"""Reproducible tiny-model microbenchmark; not a large-model speed prediction.
Run from the project folder: python benchmarks/compare_speed.py
"""
import json,statistics,sys,tempfile,threading,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from transformers import LlamaConfig,LlamaForCausalLM
from pocketllm.streaming_runtime import load_skeleton
from pocketllm.planner import GIB

def main():
    torch.set_num_threads(2);torch.manual_seed(42)
    results={}
    with tempfile.TemporaryDirectory() as directory:
        cfg=LlamaConfig(vocab_size=512,hidden_size=128,intermediate_size=256,num_hidden_layers=4,
                        num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=512)
        reference=LlamaForCausalLM(cfg);reference.save_pretrained(directory);del reference
        tokens=torch.randint(0,512,(1,128))
        outputs={}
        for label,batch,cache_budget in [('strict',32,0),('smart',256,512*1024)]:
            runs=[];reads=[]
            for _ in range(3):
                model,pager,info=load_skeleton(directory,dict(ram_cap=4*GIB,vram_cap=0,
                    gpu_enabled=False,weight_cache_budget=cache_budget),threading.Event())
                start=time.perf_counter();kv=None
                with torch.inference_mode():
                    for offset in range(0,tokens.shape[-1],batch):
                        out=model(tokens[:,offset:offset+batch],past_key_values=kv,use_cache=True,logits_to_keep=1)
                        kv=out.past_key_values
                    first=time.perf_counter()-start
                    for i in range(8):
                        out=model(torch.tensor([[i+1]]),past_key_values=kv,use_cache=True,logits_to_keep=1)
                        kv=out.past_key_values
                runs.append((first,time.perf_counter()-start));reads.append(pager.disk_bytes)
                outputs[label]=out.logits.detach().clone()
                pager.release_all();del model,out,kv,pager
            results[label]={'prefill_seconds_median':statistics.median(r[0] for r in runs),
                            'total_seconds_median':statistics.median(r[1] for r in runs),
                            'logical_weight_bytes_read_median':statistics.median(reads),
                            'prefill_batch':batch,'weight_cache_bytes':cache_budget}
        torch.testing.assert_close(outputs['strict'],outputs['smart'],rtol=1e-5,atol=1e-6)
    results['note']='Tiny random Llama; 128 prompt tokens + 8 decode passes; CPU; OS file cache warm. No claim about large models or user hardware.'
    print(json.dumps(results,indent=2))
if __name__=='__main__':main()
