import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pocketllm.hardware import engine_devices, snapshot
from pocketllm.planner import plan, GIB, component_fits_gpu

class GPUSelectionTests(unittest.TestCase):
    def devices(self):
        return [dict(id='Vulkan0',name='Intel UHD',kind='integrated / shared',backend='Vulkan',total=8*GIB,free=6*GIB,used=2*GIB,uuid='',ready=True,telemetry=False),dict(id='Vulkan1',name='AMD Radeon',kind='dedicated',backend='Vulkan',total=2*GIB,free=GIB,used=GIB,uuid='',ready=True,telemetry=False)]
    @patch('pocketllm.hardware.display_adapters',return_value=[])
    @patch('pocketllm.hardware.nv',None)
    def test_select_each_gpu_and_cpu(self,_):
        for gpu in self.devices():
            hw=snapshot(gpu['id'],devices=self.devices())
            self.assertEqual(hw['gpu_name'],gpu['name']);self.assertEqual(hw['gpu_total'],gpu['total'])
        self.assertFalse(snapshot('none',devices=self.devices())['gpu_ready'])
        with self.assertRaises(ValueError):snapshot('missing',devices=self.devices())
    @patch('pocketllm.hardware.resolve_engine',return_value='llama-server')
    @patch('subprocess.run')
    def test_engine_inventory_parses_both_adapters(self,run,_):
        run.return_value=SimpleNamespace(stdout='Available devices:\n Vulkan0: Intel(R) UHD Graphics 620 (8099 MiB, 7438 MiB free)\n Vulkan1: AMD Radeon 550X (2048 MiB, 1723 MiB free)\n')
        devices=engine_devices();self.assertEqual(len(devices),2)
        self.assertEqual(devices[0]['kind'],'integrated / shared');self.assertEqual(devices[1]['kind'],'dedicated')
    def test_shared_gpu_does_not_reduce_ram_estimate(self):
        hw=dict(ram_total=16*GIB,ram_available=12*GIB,app_rss=.2*GIB,gpu_total=8*GIB,gpu_free=6*GIB,gpu_ready=True,gpu_shared=True)
        model=dict(layers=4,block_bytes=[GIB//2]*4,size=3*GIB,kv_per_token=4096,context_max=8192)
        cpu=plan(model,hw,8,0,512);gpu=plan(model,hw,8,2,512)
        self.assertGreater(gpu['gpu_layers'],0);self.assertEqual(cpu['ram_estimate'],gpu['ram_estimate'])
    def test_low_vram_automatically_keeps_layers_in_ram(self):
        hw=dict(ram_total=16*GIB,ram_available=12*GIB,app_rss=.2*GIB,gpu_total=2*GIB,
                gpu_free=2*GIB,gpu_ready=True,gpu_id='Vulkan1')
        model=dict(layers=4,block_bytes=[GIB//2]*4,size=3*GIB,kv_per_token=4096,context_max=8192)
        allocation=plan(model,hw,8,.2,512)
        self.assertTrue(allocation['fits']);self.assertTrue(allocation['gpu_fallback'])
        self.assertEqual(allocation['gpu_layers'],0);self.assertEqual(allocation['cpu_layers'],4)
        self.assertEqual(allocation['vram_estimate'],0)
        hybrid=plan(model,hw,8,1,512)
        self.assertEqual(hybrid['gpu_layers'],1);self.assertEqual(hybrid['cpu_layers'],3)
        self.assertLess(hybrid['ram_estimate'],allocation['ram_estimate'])
    def test_component_placement_changes_with_size_and_live_headroom(self):
        mib=1024**2
        # A budget below the old 512 MiB cutoff can still run small operations.
        self.assertTrue(component_fits_gpu(32*mib,1*mib,0,GIB,400*mib))
        self.assertFalse(component_fits_gpu(200*mib,1*mib,0,GIB,400*mib))
        self.assertFalse(component_fits_gpu(32*mib,1*mib,128*mib,GIB,400*mib))
        self.assertFalse(component_fits_gpu(32*mib,1*mib,0,400*mib,400*mib))
        self.assertFalse(component_fits_gpu(32*mib,64*mib,0,GIB,400*mib))
    @patch('pocketllm.hardware.display_adapters',return_value=[])
    @patch('pocketllm.hardware.nv',None)
    def test_streaming_cannot_claim_vulkan_support(self,_):
        self.assertFalse(snapshot('Vulkan0',mode='streaming',devices=self.devices())['gpu_ready'])

if __name__=='__main__':unittest.main()
