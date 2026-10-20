import collections
import json
import http.client
import os
import re
import secrets
import shutil
import socket
import sys
import importlib.util
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
from .hardware import snapshot, engine_devices, resolve_engine
from .gguf import inspect_model
from .planner import plan, GIB
from .streaming_model import inspect_streaming, plan_streaming, public_model

class Engine:
    def __init__(self):
        self.lock = threading.RLock()
        self.chat_lock = threading.Lock()
        self.cancel = threading.Event()
        self.proc = None
        self.status = 'idle'
        self.error = ''
        self.model = None
        self.allocation = None
        self.actual_layers = None
        self.logs = collections.deque(maxlen=100)
        self.key = secrets.token_urlsafe(32)
        self.port = 0
        self.connection = None
        self.mode = "resident"
        self.paging = None
        self.closed = False
        self.gpu_baseline = 0
        self.gpu_id = None
        self.devices = []
        threading.Thread(target=self.monitor,daemon=True).start()

    def state(self):
        with self.lock:
            return dict(status=self.status,error=self.error,model=self.model,plan=self.allocation,
                        actual_gpu_layers=self.actual_layers,logs=list(self.logs),mode=self.mode,paging=self.paging)

    def cancel_generation(self):
        self.cancel.set()
        if self.mode == "streaming" and self.proc:
            try:
                with urllib.request.urlopen(self.request("/cancel",{}),timeout=1):
                    pass
            except Exception:
                pass
        connection = self.connection
        if connection and connection.sock:
            try:
                connection.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def stop(self, error=''):
        with self.lock:
            self.cancel_generation()
            p, self.proc = self.proc, None
            if p and p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait(timeout=3)
            self.paging = None
            self.status = 'error' if error else 'idle'
            self.error = error
            self.actual_layers = None

    def launch(self, path, executable, ram_gib, vram_gib, context, mode="resident", profile="smart", gpu_id=None):
        if mode == "streaming":
            return self.launch_streaming(path,ram_gib,vram_gib,context,profile,gpu_id)
        if mode != "resident":
            raise ValueError("Unknown loading mode.")
        with self.lock:
            if self.proc or self.chat_lock.locked():
                raise ValueError('Unload the current model and wait for the response to finish first.')
            self.mode = "resident"
            self.paging = None
            resolved = resolve_engine(executable)
            if not resolved:
                raise ValueError('llama-server was not found. Select its executable in Engine setup.')
            model = inspect_model(path)
            self.devices = engine_devices(resolved)
            self.gpu_id = gpu_id
            hw = snapshot(gpu_id, devices=self.devices) 
            self.gpu_id = hw['gpu_id']
            allocation = plan(model,hw,ram_gib,vram_gib,context)
            if not allocation['fits']:
                raise ValueError(allocation['reason'])
            env = os.environ.copy()
            # Do not inherit llama argument overrides from the user's environment.
            env = {k:v for k,v in env.items() if not k.startswith('LLAMA_ARG_')}
            if hw['gpu_uuid']:
                env['CUDA_VISIBLE_DEVICES'] = hw['gpu_uuid']
            flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            try:
                help_result = subprocess.run([resolved,'--help'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                             timeout=15,env=env,creationflags=flags,text=True,errors='replace')
                help_text = help_result.stdout
            except Exception as exc:
                raise ValueError(f'Cannot run llama-server: {exc}') from exc
            required = ['--fit','--load-mode','--no-kv-offload','--no-repack','--no-op-offload']
            if any(flag not in help_text for flag in required):
                raise ValueError('This llama-server build is too old. Install a current build with --fit and --load-mode support.')
            if allocation['gpu_layers']:
                devices = subprocess.run([resolved,'--list-devices'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                         timeout=15,env=env,creationflags=flags,text=True,errors='replace').stdout
                device = 'CUDA0' if hw['gpu_uuid'] else hw['gpu_id']
                if not re.search(r'\b'+re.escape(device)+r'\b',devices):
                    raise ValueError('Selected GPU is not supported by this engine. Use a matching CUDA/Vulkan llama.cpp build or select CPU.')
            with socket.socket() as s:
                s.bind(('127.0.0.1',0))
                self.port = s.getsockname()[1]
            args = [resolved,'--model',model['path'],'--host','127.0.0.1','--port',str(self.port),
                    '--api-key',self.key,'--ctx-size',str(context),'--parallel','1',
                    '--batch-size','128','--ubatch-size','128','--threads',str(max(1,(os.cpu_count() or 2)-1)),
                    '--gpu-layers',str(allocation['gpu_layers']),'--split-mode','none','--fit','off',
                    '--device',('CUDA0' if hw['gpu_uuid'] else hw['gpu_id']) if allocation['gpu_layers'] else 'none',
                    '--no-kv-offload','--no-op-offload','--no-repack','--load-mode','mmap']
            self.model,self.allocation = model,allocation
            self.actual_layers = None
            self.logs.clear()
            self.error = ''
            self.cancel.clear()
            self.gpu_baseline = hw['gpu_used']
            self.proc = subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                         text=True,errors='replace',bufsize=1,env=env,creationflags=flags)
            self.status = 'loading'
            proc = self.proc
            threading.Thread(target=self.read_logs,args=(proc,),daemon=True).start()
            threading.Thread(target=self.await_ready,args=(proc,),daemon=True).start()
            return allocation

    def launch_streaming(self,path,ram_gib,vram_gib,context,profile="smart", gpu_id=None):
        with self.lock:
            if self.proc or self.chat_lock.locked():
                raise ValueError('Unload the current model and wait for generation to finish.')
            missing=[name for name in ('torch','transformers','accelerate','safetensors') if importlib.util.find_spec(name) is None]
            if missing:
                raise ValueError('On-demand dependencies are missing. Install requirements-streaming.txt as described in README.md.')
            model=inspect_streaming(path)
            self.gpu_id=gpu_id;self.devices=[]
            hw=snapshot(gpu_id,mode='streaming')
            self.gpu_id=hw['gpu_id']
            if vram_gib and not hw['gpu_ready']:raise ValueError(hw['gpu_note'])
            if vram_gib:
                import torch
                if not torch.cuda.is_available():raise ValueError('CUDA is unavailable in PyTorch. Install its CUDA build or select CPU.')
            allocation=plan_streaming(model,hw,ram_gib,vram_gib,context,profile)
            if not allocation['fits']:raise ValueError(allocation['reason'])
            with socket.socket() as sock:
                sock.bind(('127.0.0.1',0));self.port=sock.getsockname()[1]
            env=os.environ.copy()
            env['POCKET_WORKER_KEY']=self.key
            env['CUDA_VISIBLE_DEVICES']=hw['gpu_uuid'] if hw['gpu_ready'] else ''
            args=[sys.executable,'-m','pocketllm.streaming_worker','--model',model['path'],
                  '--port',str(self.port),'--context',str(context),
                  '--ram-cap',str(allocation['ram_cap']),'--vram-cap',str(allocation['gpu_work_budget']),
                  '--weight-cache',str(allocation['weight_cache_budget']),'--prefill-batch',str(allocation['prefill_batch']),
                  '--profile',profile]
            self.mode='streaming';self.paging=None
            self.model=public_model(model);self.allocation=allocation
            self.actual_layers=None;self.logs.clear();self.error='';self.cancel.clear()
            self.gpu_baseline=hw['gpu_used']
            flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
            self.proc=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
                                       errors='replace',bufsize=1,env=env,creationflags=flags,
                                       cwd=str(Path(__file__).resolve().parent.parent))
            self.status='loading';proc=self.proc
            threading.Thread(target=self.read_logs,args=(proc,),daemon=True).start()
            threading.Thread(target=self.await_ready,args=(proc,),daemon=True).start()
            return allocation

    def read_logs(self, proc):
        for line in proc.stdout:
            with self.lock:
                if self.proc is not proc:
                    break
                if line.startswith("POCKET_EVENT "):
                    try:
                        self.paging=json.loads(line[len("POCKET_EVENT "):])
                    except ValueError:
                        pass
                    continue
                self.logs.append(line.strip())
                match = re.search(r'offloaded (\d+)/(\d+) layers',line)
                if match:
                    self.actual_layers = int(match[1])
        proc.stdout.close()

    def await_ready(self,proc):
        deadline = time.monotonic()+600
        while time.monotonic() < deadline:
            with self.lock:
                if self.proc is not proc:
                    return
            try:
                with urllib.request.urlopen(self.request('/health'),timeout=2) as response:
                    if response.status == 200:
                        with self.lock:
                            if self.proc is proc:
                                if self.actual_layers is not None and self.actual_layers != self.allocation['gpu_layers']:
                                    self.stop('Engine offload differs from the memory plan. Model unloaded.')
                                else:
                                    self.status = 'ready'
                        return
            except Exception:
                pass
            time.sleep(0.4)
        with self.lock:
            if self.proc is proc:
                self.stop('Model loading exceeded 10 minutes. Check engine logs.')

    def request(self,path,payload=None):
        return urllib.request.Request(f'http://127.0.0.1:{self.port}{path}',
                    data=json.dumps(payload).encode() if payload is not None else None,
                    headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json'})

    def monitor(self):
        while not self.closed:
            time.sleep(0.35)
            with self.lock:
                if not self.proc:
                    continue
                if self.proc.poll() is not None:
                    code = self.proc.returncode
                    self.stop(f'Inference process exited ({code}). Review logs; reduce budgets/context or check engine compatibility.')
                    continue
                try:
                    hw = snapshot(self.gpu_id, devices=self.devices, mode=self.mode)
                except Exception:
                    self.stop('Hardware monitoring failed. Model unloaded.')
                    continue
                p = self.allocation
                if hw['app_rss'] > p['ram_cap']:
                    self.stop('RAM budget reached. Model unloaded; try a smaller model or context.')
                elif hw['ram_available'] < 0.5*GIB:
                    self.stop('System RAM is under pressure. Model unloaded to preserve headroom.')
                elif (p['gpu_layers'] or p.get('gpu_enabled')) and not hw['gpu_ready']:
                    self.stop('GPU telemetry was lost. Model unloaded.')
                elif (p['gpu_layers'] or p.get('gpu_enabled')) and hw.get('gpu_telemetry',True) and (hw['gpu_used']-self.gpu_baseline > p['vram_cap'] or hw['gpu_free'] < 0.125*GIB):
                    self.stop('GPU memory budget or system headroom reached. Model unloaded.')

    def stream(self,messages):
        if not self.chat_lock.acquire(blocking=False):
            raise ValueError('A response is already running.')
        try:
            with self.lock:
                if self.status != 'ready':
                    raise ValueError('Load a model before chatting.')
                self.cancel.clear()
                proc = self.proc
            # Ask the engine to tokenize the templated conversation before generation.
            with urllib.request.urlopen(self.request('/apply-template',{'messages':messages}),timeout=30) as r:
                prompt = json.load(r)['prompt']
            with urllib.request.urlopen(self.request('/tokenize',{'content':prompt,'add_special':True}),timeout=30) as r:
                count = len(json.load(r)['tokens'])
            remaining = self.allocation['context']-count-16
            if remaining < 32:
                raise ValueError('Conversation fills the context window. Start a new chat or reload with a larger context.')
            data = {'messages':messages,'stream':True,'max_tokens':min(512,remaining),'temperature':0.7}
            started, tokens = time.monotonic(),0
            if self.cancel.is_set() or self.proc is not proc:
                return
            connection = http.client.HTTPConnection('127.0.0.1',self.port,timeout=180)
            self.connection = connection
            connection.request('POST','/v1/chat/completions',body=json.dumps(data),
                               headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json'})
            with connection.getresponse() as response:
                if response.status != 200:
                    raise ValueError(response.read(4096).decode(errors='replace'))
                for line in response:
                    if self.cancel.is_set() or self.proc is not proc:
                        break
                    if not line.startswith(b'data: '):
                        continue
                    body = line[6:].strip()
                    if body == b'[DONE]':
                        break
                    chunk = json.loads(body)
                    if 'error' in chunk:
                        raise ValueError(str(chunk['error']))
                    choices = chunk.get('choices',[])
                    if choices:
                        delta = choices[0].get('delta',{})
                        text = delta.get('content') or delta.get('reasoning_content') or ''
                        if text:
                            tokens += 1
                            yield {'type':'token','text':text}
            yield {'type':'done','seconds':round(time.monotonic()-started,2),'chunks':tokens}
        finally:
            if self.connection:
                self.connection.close()
                self.connection = None
            self.chat_lock.release()
