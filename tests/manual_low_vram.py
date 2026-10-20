import time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pocketllm.engine import Engine
engine=Engine()
try:
    allocation=engine.launch(str(Path('../models/qwen2.5-3b.gguf').resolve()),'',6,.1,512,gpu_id='Vulkan1')
    print('Low-VRAM plan:',allocation['gpu_layers'],'GPU layers;',allocation['cpu_layers'],'RAM layers; fallback:',allocation['gpu_fallback'],flush=True)
    deadline=time.monotonic()+90
    while engine.status=='loading' and time.monotonic()<deadline:time.sleep(.5)
    if engine.status!='ready':raise RuntimeError(engine.error+'\n'+'\n'.join(engine.logs))
    answer=''.join(event['text'] for event in engine.stream([{'role':'user','content':'Reply with one word: hello.'}]) if event['type']=='token')
    print('Response:',answer,flush=True)
finally:
    engine.closed=True;engine.stop()
