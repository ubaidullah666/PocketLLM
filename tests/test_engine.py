import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from test_core import fixture
from pocketllm.engine import Engine
from pocketllm.planner import GIB

@unittest.skipIf(os.name=='nt','POSIX executable fixture; production supports Windows native engines')
class EngineTests(unittest.TestCase):
    def setUp(self):
        self.engine=Engine()
        self.temp=tempfile.TemporaryDirectory()
        self.model=fixture(Path(self.temp.name)/'fixture.gguf')
        self.exe=Path(__file__).with_name('fake_engine.py').resolve();self.exe.chmod(0o755)
    def tearDown(self):
        self.engine.closed=True;self.engine.stop();self.temp.cleanup()
    def load(self):
        self.engine.launch(str(self.model),str(self.exe),4,0,512)
        deadline=time.monotonic()+5
        while self.engine.status=='loading' and time.monotonic()<deadline:time.sleep(.05)
        self.assertEqual(self.engine.status,'ready',self.engine.error)
    def test_load_stream_unload(self):
        self.load()
        self.assertEqual(self.engine.actual_layers,0)
        events=list(self.engine.stream([{'role':'user','content':'hello'}]))
        self.assertEqual(''.join(e.get('text','') for e in events),'Hello from fixture')
        self.assertEqual(events[-1]['type'],'done')
        self.engine.stop();self.assertEqual(self.engine.status,'idle')
        self.assertIsNone(self.engine.proc)
    def test_memory_watchdog_unloads(self):
        self.load()
        from pocketllm.hardware import snapshot
        hw=snapshot();hw['app_rss']=5*GIB
        with patch('pocketllm.engine.snapshot',return_value=hw):
            deadline=time.monotonic()+3
            while self.engine.status=='ready' and time.monotonic()<deadline:time.sleep(.05)
        self.assertEqual(self.engine.status,'error');self.assertIn('RAM budget',self.engine.error)
    def test_child_failure_is_contained(self):
        self.load();self.engine.proc.kill()
        deadline=time.monotonic()+3
        while self.engine.status=='ready' and time.monotonic()<deadline:time.sleep(.05)
        self.assertEqual(self.engine.status,'error')
    def test_rejects_second_generation(self):
        self.load();self.engine.chat_lock.acquire()
        try:
            with self.assertRaises(ValueError):list(self.engine.stream([{'role':'user','content':'hi'}]))
        finally:self.engine.chat_lock.release()
