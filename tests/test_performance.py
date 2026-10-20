import threading
import tempfile
import unittest
from pocketllm.performance import WeightCache,PrefixCache
from pocketllm.planner import GIB
from test_streaming import DEPS

class CacheTests(unittest.TestCase):
    def test_bounded_cache_avoids_cyclic_thrashing(self):
        c=WeightCache(20)
        for key in ('a','b','c'):c.admit(key,key,10)
        self.assertEqual(c.bytes,20)
        self.assertEqual(c.get('a'),'a');self.assertEqual(c.get('b'),'b')
        self.assertIsNone(c.get('c'))
        self.assertFalse(c.admit('c','c',10));self.assertEqual(c.get('a'),'a')
    def test_pressure_trim_releases_storage(self):
        c=WeightCache(30)
        for k in range(3):c.admit(k,str(k),10)
        c.trim(10);self.assertLessEqual(c.bytes,10)
        c.trim();self.assertEqual(c.bytes,0);self.assertFalse(c.items)
    def test_prefix_requires_exact_match_and_is_consumed(self):
        c=PrefixCache();state=object();c.save([1,2],state)
        self.assertEqual(c.take([1,2,3]),(state,2));self.assertEqual(c.take([1,2,3]),(None,0))
        c.save([1,2],state);self.assertEqual(c.take([1,4,3]),(None,0))
        c.save([1,2],state);self.assertEqual(c.take([1,2]),(None,0))

@unittest.skipUnless(DEPS,'Streaming dependencies required')
class RealCacheTests(unittest.TestCase):
    def test_reuse_reduces_reads_without_changing_outputs(self):
        import torch
        from transformers import LlamaConfig,LlamaForCausalLM
        from pocketllm.streaming_runtime import load_skeleton
        with tempfile.TemporaryDirectory() as d:
            config=LlamaConfig(vocab_size=128,hidden_size=64,intermediate_size=128,num_hidden_layers=2,
                num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=512)
            config._attn_implementation='sdpa';ref=LlamaForCausalLM(config).eval();ref.save_pretrained(d)
            alloc=dict(ram_cap=4*GIB,vram_cap=0,gpu_enabled=False,weight_cache_budget=65536)
            model,pager,_=load_skeleton(d,alloc,threading.Event())
            ids=torch.tensor([[1,2,3]])
            with torch.inference_mode():
                baseline=ref(ids).logits
                torch.testing.assert_close(model(ids).logits,baseline,rtol=1e-5,atol=1e-6)
                reads=pager.disk_bytes
                pager.release_all(keep_cache=True)
                torch.testing.assert_close(model(ids).logits,baseline,rtol=1e-5,atol=1e-6)
            self.assertLess(pager.disk_bytes-reads,reads)
            self.assertGreater(pager.cache.hits,0)
            self.assertLessEqual(pager.cache.bytes,65536)
            self.assertEqual(pager.resident_bytes,0)
            pager.release_all();self.assertEqual(pager.cache.bytes,0)
    def test_prefix_reuse_matches_full_prompt_logits(self):
        import torch
        from transformers import LlamaConfig,LlamaForCausalLM
        config=LlamaConfig(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
                num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=512)
        config._attn_implementation='sdpa';model=LlamaForCausalLM(config).eval()
        c=PrefixCache()
        with torch.inference_mode():
            first=model(torch.tensor([[1,2,3]]),use_cache=True)
            c.save([1,2,3],first.past_key_values)
            state,reused=c.take([1,2,3,4,5])
            actual=model(torch.tensor([[4,5]]),past_key_values=state,use_cache=True).logits[:,-1]
            expected=model(torch.tensor([[1,2,3,4,5]])).logits[:,-1]
            torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
            self.assertEqual(reused,3)
