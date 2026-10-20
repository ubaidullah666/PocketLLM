"""Isolated local HTTP worker for on-demand inference. No network model downloads."""
import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer

# Prevent Transformers from fetching models or sending telemetry.
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True);parser.add_argument('--port',type=int,required=True)
    parser.add_argument('--context',type=int,required=True)
    parser.add_argument('--ram-cap',type=int,required=True);parser.add_argument('--vram-cap',type=int,required=True)
    parser.add_argument('--weight-cache',type=int,default=0)
    parser.add_argument('--prefill-batch',type=int,default=32)
    parser.add_argument('--profile',choices=['smart','strict'],default='strict')
    args=parser.parse_args()
    import torch
    from transformers import AutoTokenizer,TextStreamer
    from .streaming_runtime import load_skeleton,Cancelled
    from .performance import PrefixCache
    prefix=PrefixCache()
    key=os.environ.pop('POCKET_WORKER_KEY')
    stop=threading.Event();generation_lock=threading.Lock()
    allocation=dict(ram_cap=args.ram_cap,vram_cap=args.vram_cap,gpu_enabled=args.vram_cap>256*1024**2,weight_cache_budget=args.weight_cache)
    def emit(event):
        print('POCKET_EVENT '+json.dumps(event),flush=True)
    model,pager,info=load_skeleton(args.model,allocation,stop,emit)
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True,trust_remote_code=False)
    if not tokenizer.chat_template:
        raise ValueError('The local tokenizer has no chat template. Use an instruct/chat checkpoint with its tokenizer configuration.')
    import psutil
    torch.set_num_threads(max(1,min(8,psutil.cpu_count(logical=False) or 2)))
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*_):pass
        def authorized(self):return self.headers.get('Authorization')=='Bearer '+key
        def reply(self,data,status=200):
            raw=json.dumps(data).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
        def do_GET(self):
            if not self.authorized():return self.reply({'error':'Unauthorized'},403)
            self.reply({'status':'ok'} if self.path=='/health' else {'error':'Not found'},200 if self.path=='/health' else 404)
        def do_POST(self):
            if not self.authorized():return self.reply({'error':'Unauthorized'},403)
            try:
                length=int(self.headers.get('Content-Length',0))
                if not 0<length<=1_000_000:raise ValueError('Invalid request size.')
                data=json.loads(self.rfile.read(length))
                if self.path=='/cancel':
                    stop.set();return self.reply({'ok':True})
                if self.path=='/apply-template':
                    prompt=tokenizer.apply_chat_template(data['messages'],tokenize=False,add_generation_prompt=True)
                    return self.reply({'prompt':prompt})
                if self.path=='/tokenize':
                    return self.reply({'tokens':tokenizer.encode(data['content'],add_special_tokens=False)})
                if self.path!='/v1/chat/completions':return self.reply({'error':'Not found'},404)
                if not generation_lock.acquire(blocking=False):return self.reply({'error':'Generation already running'},409)
            except Exception as exc:
                return self.reply({'error':str(exc)},400)
            cache=None;out=None;started=False;completed=False
            request_started=time.monotonic();generated=0;first_token=None
            heartbeat_done=threading.Event();write_lock=threading.Lock()
            def send(raw):
                with write_lock:
                    self.wfile.write(raw);self.wfile.flush()
            def heartbeat():
                while not heartbeat_done.wait(5):
                    try:send(b": working\n\n")
                    except OSError:
                        stop.set();return
            try:
                stop.clear()
                pager.begin_request()
                encoded=tokenizer.apply_chat_template(data['messages'],tokenize=True,add_generation_prompt=True,return_tensors='pt')
                token_ids=encoded[0].tolist()
                cache,reused=prefix.take(token_ids) if args.profile=='smart' else (None,0)
                processed_ids=token_ids[:reused]
                pager.metrics=dict(reused_prompt_tokens=reused,generated_tokens=0,first_token_ms=None,tokens_per_second=0)
                remaining=args.context-encoded.shape[-1]-16
                limit=min(int(data.get('max_tokens',512)),remaining)
                if remaining<1:raise ValueError('Context window is full. Start a new chat.')
                self.send_response(200);self.send_header('Content-Type','text/event-stream');self.send_header('Cache-Control','no-cache');self.end_headers()
                started=True
                threading.Thread(target=heartbeat,daemon=True).start()
                def chunk(text):
                    payload={'choices':[{'delta':{'content':text}}]}
                    send(('data: '+json.dumps(payload)+'\n\n').encode())
                class Streamer(TextStreamer):
                    def on_finalized_text(self,text,stream_end=False):
                        if text:chunk(text)
                streamer=Streamer(tokenizer,skip_prompt=False,skip_special_tokens=True)
                eos=model.generation_config.eos_token_id or tokenizer.eos_token_id
                eos=set(eos if isinstance(eos,list) else [eos])
                with torch.inference_mode():
                    # Small prefill chunks bound temporary activations. Every
                    # forward uses the same load/use/release component hooks.
                    for start in range(reused,encoded.shape[-1],args.prefill_batch):
                        if stop.is_set():raise Cancelled()
                        out=model(input_ids=encoded[:,start:start+args.prefill_batch],past_key_values=cache,use_cache=True,logits_to_keep=1)
                        cache=out.past_key_values
                        processed_ids=token_ids[:min(encoded.shape[-1],start+args.prefill_batch)]
                        send(b': prefill\n\n')
                    for index in range(limit):
                        if stop.is_set():raise Cancelled()
                        next_token=out.logits[:,-1,:].argmax(dim=-1)
                        if next_token.item() in eos:break
                        generated+=1
                        if first_token is None:first_token=time.monotonic()
                        streamer.put(next_token)
                        pager.metrics.update(generated_tokens=generated,first_token_ms=round((first_token-request_started)*1000,1),
                            tokens_per_second=round((generated-1)/max(.001,time.monotonic()-first_token),2))
                        pager.report(force=True)
                        if index+1==limit:break  # No unused final forward pass.
                        processed_ids.append(next_token.item())
                        del out
                        out=model(input_ids=next_token[:,None],past_key_values=cache,use_cache=True,logits_to_keep=1)
                        cache=out.past_key_values
                    streamer.end()
                if args.profile=='smart':prefix.save(processed_ids,cache)
                completed=True
                send(b'data: [DONE]\n\n')
            except Cancelled:
                pass
            except (BrokenPipeError,ConnectionResetError):
                stop.set()
            except Exception as exc:
                try:
                    if started:send(('data: '+json.dumps({'error':str(exc)})+'\n\n').encode())
                    else:self.reply({'error':str(exc)},400)
                except OSError:pass
            finally:
                heartbeat_done.set()
                cache=None;out=None
                if not completed:prefix.clear()
                pager.release_all(keep_cache=completed and args.profile=='smart')
                generation_lock.release()
    class Server(ThreadingHTTPServer):
        daemon_threads=True
    print('On-demand engine ready: weights stay on disk until a component executes.',flush=True)
    Server(('127.0.0.1',args.port),Handler).serve_forever()

if __name__=='__main__':main()
