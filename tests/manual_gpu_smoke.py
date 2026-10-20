import time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pocketllm.engine import Engine
from pocketllm.hardware import engine_devices, snapshot
print(engine_devices(),flush=True)
for gpu in ('Vulkan0','Vulkan1'):
 e=Engine()
 try:
  hw=snapshot(gpu,devices=engine_devices())
  print('Testing',gpu,hw['gpu_name'],flush=True)
  p=e.launch(str(Path('../models/qwen2.5-3b.gguf').resolve()),'',6,1,512,gpu_id=gpu)
  deadline=time.monotonic()+120
  while e.status=='loading' and time.monotonic()<deadline:time.sleep(.5)
  print(gpu,e.status,p['gpu_layers'],e.actual_layers,e.error,flush=True)
  if e.status!='ready':print('\n'.join(e.logs),flush=True);raise RuntimeError('GPU load failed')
  chunks=[]
  for event in e.stream([{'role':'user','content':'Reply with one word: hello.'}]):
   if event['type']=='token': chunks.append(event['text'])
  print('Response', ''.join(chunks),flush=True)
 finally:e.closed=True;e.stop()
