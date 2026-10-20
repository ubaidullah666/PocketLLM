import importlib.util
import json
import struct
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from pocketllm.streaming_model import inspect_streaming,plan_streaming
from pocketllm.planner import GIB

DEPS=all(importlib.util.find_spec(n) for n in ('torch','transformers','accelerate','safetensors'))

class HeaderTests(unittest.TestCase):
    def test_does_not_require_whole_model_in_ram(self):
        hw=dict(ram_total=16*GIB,ram_available=10*GIB,app_rss=.2*GIB,gpu_total=0,gpu_free=0,gpu_ready=False)
        model=dict(size=100*GIB,layers=40,context_max=4096,kv_per_token=16384,
                   largest_component=256*1024**2,largest_source=128*1024**2,
                   hidden_size=4096,heads=32,vocab_size=32000)
        p=plan_streaming(model,hw,4,0,512)
        self.assertTrue(p['fits']);self.assertEqual(p['disk_layers'],40)
        self.assertEqual(p['cpu_layers'],0);self.assertEqual(p['gpu_layers'],0)
    def test_largest_component_must_fit(self):
        hw=dict(ram_total=16*GIB,ram_available=10*GIB,app_rss=.2*GIB,gpu_total=0,gpu_free=0,gpu_ready=False)
        model=dict(size=100*GIB,layers=40,context_max=4096,kv_per_token=16384,
                   largest_component=8*GIB,largest_source=4*GIB,
                   hidden_size=4096,heads=32,vocab_size=32000)
        self.assertFalse(plan_streaming(model,hw,4,0,512)['fits'])
    def test_rejects_quantized_model(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'config.json').write_text(json.dumps({'model_type':'llama','quantization_config':{'bits':4}}))
            with self.assertRaises(ValueError):inspect_streaming(d)

@unittest.skipUnless(DEPS,'Optional streaming dependencies not installed')
class PagingTests(unittest.TestCase):
    def setUp(self):
        import torch
        from transformers import LlamaConfig,LlamaForCausalLM
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        torch.manual_seed(123)
        self.reference=LlamaForCausalLM(LlamaConfig(vocab_size=64,hidden_size=32,intermediate_size=64,
            num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=512)).eval()
        self.reference.config._attn_implementation='sdpa'
        self.reference.save_pretrained(self.root,safe_serialization=True,max_shard_size='10KB')
        self.allocation=dict(ram_cap=4*GIB,vram_cap=0,gpu_enabled=False)
    def tearDown(self):self.temp.cleanup()
    def test_real_model_matches_resident_logits_and_releases_all_weights(self):
        import torch
        from pocketllm.streaming_runtime import load_skeleton
        streamed,pager,info=load_skeleton(self.root,self.allocation,threading.Event())
        self.assertTrue(all(p.device.type=='meta' for p in streamed.parameters()))
        self.assertEqual(pager.loads,0)
        ids=torch.tensor([[1,2,3,4]])
        with torch.inference_mode():
            expected=self.reference(ids,use_cache=True)
            actual=streamed(ids,use_cache=True)
            torch.testing.assert_close(actual.logits,expected.logits,rtol=1e-5,atol=1e-6)
            # Repeat with a live KV cache to verify the next autoregressive token.
            expected2=self.reference(torch.tensor([[5]]),past_key_values=expected.past_key_values,use_cache=True)
            actual2=streamed(torch.tensor([[5]]),past_key_values=actual.past_key_values,use_cache=True)
            torch.testing.assert_close(actual2.logits,expected2.logits,rtol=1e-5,atol=1e-6)
        self.assertTrue(all(p.device.type=='meta' for p in streamed.parameters()))
        self.assertEqual(pager.resident_bytes,0);self.assertEqual(pager.loads,pager.releases)
        self.assertLessEqual(pager.peak_bytes,info['largest_component'])
        self.assertGreater(pager.loads,info['layers']*2)
    def test_cancel_before_read(self):
        import torch
        from pocketllm.streaming_runtime import load_skeleton,Cancelled
        cancel=threading.Event();streamed,pager,_=load_skeleton(self.root,self.allocation,cancel)
        cancel.set()
        with torch.inference_mode(),self.assertRaises(Cancelled):streamed(torch.tensor([[1]]))
        self.assertEqual(pager.loads,0);self.assertTrue(all(p.device.type=='meta' for p in streamed.parameters()))
    def test_budget_refusal_does_not_load_weights(self):
        import torch
        from pocketllm.streaming_runtime import load_skeleton
        allocation={**self.allocation,'ram_cap':1}
        streamed,pager,_=load_skeleton(self.root,allocation,threading.Event())
        with torch.inference_mode(),self.assertRaises(MemoryError):streamed(torch.tensor([[1]]))
        self.assertEqual(pager.loads,0)
    def test_changed_file_rejected(self):
        import torch
        from pocketllm.streaming_runtime import load_skeleton
        streamed,pager,info=load_skeleton(self.root,self.allocation,threading.Event())
        first=info['entries']['model.embed_tokens.weight']['file']
        import os
        os.utime(first,ns=(1,1))
        with torch.inference_mode(),self.assertRaises(ValueError):streamed(torch.tensor([[1]]))
        self.assertEqual(pager.resident_bytes,0)
    def test_release_after_forward_exception(self):
        import torch
        from pocketllm.streaming_runtime import load_skeleton
        streamed,pager,_=load_skeleton(self.root,self.allocation,threading.Event())
        with patch.object(streamed.model.embed_tokens,'forward',side_effect=RuntimeError('test failure')):
            with torch.inference_mode(),self.assertRaises(RuntimeError):streamed(torch.tensor([[1]]))
        self.assertEqual(pager.resident_bytes,0)
        self.assertTrue(all(p.device.type=='meta' for p in streamed.parameters()))


@unittest.skipUnless(DEPS,'Optional streaming dependencies not installed')
class OtherArchitectures(unittest.TestCase):
    def test_qwen_dense_models_match(self):
        import torch
        from transformers import Qwen2Config,Qwen2ForCausalLM,Qwen3Config,Qwen3ForCausalLM
        from pocketllm.streaming_runtime import load_skeleton
        for config_cls,model_cls in [(Qwen2Config,Qwen2ForCausalLM),(Qwen3Config,Qwen3ForCausalLM)]:
            with self.subTest(model=model_cls.__name__),tempfile.TemporaryDirectory() as d:
                config=config_cls(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
                                  num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=512)
                config._attn_implementation='sdpa'
                ref=model_cls(config).eval();ref.save_pretrained(d)
                model,pager,_=load_skeleton(d,dict(ram_cap=4*GIB,vram_cap=0,gpu_enabled=False),threading.Event())
                with torch.inference_mode():
                    ids=torch.tensor([[1,2,3]])
                    torch.testing.assert_close(model(ids).logits,ref(ids).logits,rtol=1e-5,atol=1e-6)
                self.assertEqual(pager.resident_bytes,0)
    def test_tied_embedding_weights(self):
        import torch
        from transformers import LlamaConfig,LlamaForCausalLM
        from pocketllm.streaming_runtime import load_skeleton
        with tempfile.TemporaryDirectory() as d:
            config=LlamaConfig(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=1,
                               num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=512,tie_word_embeddings=True)
            config._attn_implementation='sdpa';ref=LlamaForCausalLM(config).eval();ref.save_pretrained(d)
            model,pager,_=load_skeleton(d,dict(ram_cap=4*GIB,vram_cap=0,gpu_enabled=False),threading.Event())
            with torch.inference_mode():
                ids=torch.tensor([[1,2,3]])
                torch.testing.assert_close(model(ids).logits,ref(ids).logits,rtol=1e-5,atol=1e-6)
            self.assertEqual(pager.resident_bytes,0)

@unittest.skipUnless(DEPS,'Optional streaming dependencies not installed')
class WorkerTests(unittest.TestCase):
    def test_actual_worker_chat_and_paging_telemetry(self):
        import time
        from transformers import LlamaConfig,LlamaForCausalLM,PreTrainedTokenizerFast
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from pocketllm.engine import Engine
        with tempfile.TemporaryDirectory() as d:
            config=LlamaConfig(vocab_size=8,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
                num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=512,eos_token_id=7)
            LlamaForCausalLM(config).save_pretrained(d)
            tok=Tokenizer(WordLevel({'[UNK]':0,'hello':1,'user':2,'assistant':3,'world':4,'a':5,'b':6,'[EOS]':7},unk_token='[UNK]'))
            tok.pre_tokenizer=Whitespace()
            fast=PreTrainedTokenizerFast(tokenizer_object=tok,unk_token='[UNK]',eos_token='[EOS]')
            fast.chat_template="{% for m in messages %}{{ m['role'] + ' ' + m['content'] + ' ' }}{% endfor %}assistant "
            fast.save_pretrained(d)
            engine=Engine()
            try:
                engine.launch(d,'',4,0,256,mode='streaming')
                deadline=time.monotonic()+45
                while engine.status=='loading' and time.monotonic()<deadline:time.sleep(.1)
                self.assertEqual(engine.status,'ready',engine.state())
                self.assertEqual(engine.paging['component_loads'],0)
                # Exercise the real streaming worker API directly with a short response.
                import urllib.request
                with urllib.request.urlopen(engine.request('/v1/chat/completions',{'messages':[{'role':'user','content':'hello'}], 'max_tokens':3}),timeout=30) as response:
                    content=response.read().decode()
                self.assertIn('[DONE]',content)
                deadline=time.monotonic()+2
                while engine.paging['component_loads']==0 and time.monotonic()<deadline:time.sleep(.05)
                self.assertGreater(engine.paging['component_loads'],0)
                self.assertEqual(engine.paging['resident_weight_bytes'],0)
            finally:
                engine.closed=True;engine.stop()

if __name__=='__main__':unittest.main()
