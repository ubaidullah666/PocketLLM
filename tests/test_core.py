import json
import struct
import tempfile
import threading
import unittest
import urllib.request
import urllib.error
from pathlib import Path
from unittest.mock import patch
from pocketllm.gguf import inspect_model, GGUFError
from pocketllm.planner import plan,GIB


def fixture(path, arch='llama', split=1):
    def string(s):
        b=s.encode();return struct.pack('<Q',len(b))+b
    meta={'general.architecture':arch,f'{arch}.block_count':2,f'{arch}.attention.head_count':4,
          f'{arch}.attention.head_count_kv':2,f'{arch}.embedding_length':128,f'{arch}.context_length':4096,'split.count':split}
    data=b'GGUF'+struct.pack('<IQQ',3,3,len(meta))
    for k,v in meta.items():
        data+=string(k)+(struct.pack('<I',8)+string(v) if isinstance(v,str) else struct.pack('<II',4,v))
    for i,n in enumerate(['blk.0.attn.weight','blk.1.attn.weight','token_embd.weight']):
        data+=string(n)+struct.pack('<IQIQ',1,8,0,i*32)
    data+=bytes((-len(data))%32)+bytes(96)
    path.write_bytes(data)
    return path

class GGUFTests(unittest.TestCase):
    def test_reads_layer_spans_and_kv(self):
        with tempfile.TemporaryDirectory() as d:
            m=inspect_model(fixture(Path(d)/'tiny.gguf'))
            self.assertEqual(m['block_bytes'],[32,32])
            self.assertEqual(m['kv_per_token'],512)
    def test_rejects_unsupported_and_split(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'tiny.gguf'
            for kwargs in [{'arch':'mamba'},{'split':2}]:
                with self.assertRaises(GGUFError):inspect_model(fixture(p,**kwargs))
    def test_truncated_header(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.gguf';p.write_bytes(b'GGUF\x03')
            with self.assertRaises(GGUFError):inspect_model(p)
    def test_invalid_offsets(self):
        with tempfile.TemporaryDirectory() as d:
            p=fixture(Path(d)/'tiny.gguf');p.write_bytes(p.read_bytes()[:-80])
            with self.assertRaises(GGUFError):inspect_model(p)

class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.hw=dict(ram_total=16*GIB,ram_available=12*GIB,app_rss=.2*GIB,gpu_total=4*GIB,gpu_free=4*GIB,gpu_ready=True)
        self.model=dict(layers=4,block_bytes=[GIB//2]*4,size=3*GIB,kv_per_token=4096,context_max=8192)
    def test_cpu_only(self):
        p=plan(self.model,self.hw,8,0,2048);self.assertEqual(p['gpu_layers'],0);self.assertTrue(p['fits'])
    def test_maximum_fitting_suffix(self):
        p=plan(self.model,self.hw,8,1.5,2048)
        self.assertEqual(p['gpu_layers'],2)
        self.assertLessEqual(p['vram_estimate'],1.5*GIB)
    def test_refuses_oversize(self):
        self.assertFalse(plan(self.model,self.hw,1,0,2048)['fits'])
    def test_uses_free_memory_not_total(self):
        self.hw['gpu_free']=.3*GIB
        self.assertEqual(plan(self.model,self.hw,8,4,2048)['gpu_layers'],0)
    def test_context_cost_and_bounds(self):
        a=plan(self.model,self.hw,8,0,512);b=plan(self.model,self.hw,8,0,4096)
        self.assertGreater(b['ram_estimate'],a['ram_estimate'])
        with self.assertRaises(ValueError):plan(self.model,self.hw,8,0,16384)
    def test_missing_gpu_telemetry(self):
        self.hw['gpu_ready']=False
        self.assertEqual(plan(self.model,self.hw,8,2,2048)['gpu_layers'],0)

class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pocketllm import app
        cls.app=app;app.engine=app.Engine()
        cls.server=app.Server(('127.0.0.1',0),app.Handler)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
        cls.base=f'http://127.0.0.1:{cls.server.server_port}'
    @classmethod
    def tearDownClass(cls):
        cls.app.engine.closed=True;cls.app.engine.stop();cls.server.shutdown();cls.server.server_close()
    def call(self,path,data=None,token=True,origin=None):
        headers={'Content-Type':'application/json'}
        if token:headers['X-Session']=self.app.TOKEN
        if origin:headers['Origin']=origin
        req=urllib.request.Request(self.base+path,data=json.dumps(data).encode() if data is not None else None,headers=headers)
        return urllib.request.urlopen(req,timeout=3)
    def test_auth_required(self):
        with self.assertRaises(urllib.error.HTTPError) as e:self.call('/api/state',token=False)
        self.assertEqual(e.exception.code,403)
    def test_cross_origin_blocked(self):
        with self.assertRaises(urllib.error.HTTPError) as e:self.call('/api/unload',{},origin='https://example.com')
        self.assertEqual(e.exception.code,403)
    def test_real_hardware_state(self):
        with self.call('/api/state') as r:data=json.load(r)
        self.assertGreater(data['hardware']['ram_total'],0)
        self.assertEqual(data['engine']['status'],'idle')
    def test_bootstrap_session(self):
        with self.call('/?session='+self.app.TOKEN) as r:html=r.read()
        self.assertIn(b'PocketLLM',html);self.assertNotIn(b'__SESSION__',html)
        with self.assertRaises(urllib.error.HTTPError):self.call('/')
    def test_model_inspection_api(self):
        with tempfile.TemporaryDirectory() as d:
            p=fixture(Path(d)/'test.gguf')
            with self.call('/api/inspect',dict(path=str(p),ram=4,vram=0,context=512)) as r:data=json.load(r)
            self.assertEqual(data['model']['layers'],2)
    def test_no_model_chat_returns_error_event(self):
        with self.call('/api/chat',{'messages':[{'role':'user','content':'hello'}]}) as r:data=r.read()
        self.assertIn(b'Load a model before chatting',data)
    def test_missing_binary(self):
        with self.assertRaises(ValueError):self.app.engine.launch('x.gguf','/nonexistent/llama-server',4,0,512)

if __name__=='__main__':unittest.main()
