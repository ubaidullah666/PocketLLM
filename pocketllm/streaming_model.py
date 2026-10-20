"""Header-only inspection for local, unquantized Safetensors model folders."""
import json
import math
import struct
from pathlib import Path
from .planner import GIB

DTYPES = {'F32':4,'F16':2,'BF16':2}

def read_json(path, limit=16*1024*1024):
    if path.stat().st_size > limit:
        raise ValueError(f'{path.name} exceeds the metadata size limit.')
    return json.loads(path.read_text(encoding='utf-8'))

def inspect_streaming(path):
    root=Path(path).expanduser().resolve(strict=True)
    if root.name=='config.json':
        root=root.parent
    if not root.is_dir():
        raise ValueError('On-demand mode needs a local model folder containing config.json, tokenizer files and .safetensors weights. GGUF is supported in Resident mode.')
    config=read_json(root/'config.json')
    arch=config.get('model_type')
    if arch not in ('llama','qwen2','qwen3') or config.get('quantization_config') or config.get('num_local_experts'):
        raise ValueError('On-demand mode supports unquantized dense Llama, Qwen2 and Qwen3 Safetensors. Quantized, MoE and custom-code models are unsupported.')
    layers=int(config.get('num_hidden_layers',0)); hidden=int(config.get('hidden_size',0))
    heads=int(config.get('num_attention_heads',0)); kv_heads=int(config.get('num_key_value_heads',heads))
    context=int(config.get('max_position_embeddings',0)); head_dim=int(config.get('head_dim',hidden//max(1,heads)))
    if not 1<=layers<=1024 or min(hidden,heads,kv_heads,context,head_dim)<=0:
        raise ValueError('Invalid model dimensions.')
    index=root/'model.safetensors.index.json'
    if index.exists():
        weight_map=read_json(index).get('weight_map',{})
        if not isinstance(weight_map,dict) or not weight_map:
            raise ValueError('Invalid Safetensors index.')
        files=sorted(set(weight_map.values()))
    elif (root/'model.safetensors').is_file():
        weight_map=None;files=['model.safetensors']
    else:
        raise ValueError('Missing model.safetensors or model.safetensors.index.json.')
    entries={};size=0
    for filename in files:
        file=(root/filename).resolve(strict=True)
        if file.parent!=root or file.suffix!='.safetensors':
            raise ValueError('Weight shards must be Safetensors files directly inside the model folder.')
        stat=file.stat();size+=stat.st_size
        with file.open('rb') as f:
            first=f.read(8)
            if len(first)!=8:
                raise ValueError('Truncated Safetensors header.')
            length=struct.unpack('<Q',first)[0]
            if not 2<=length<=64*1024*1024 or length+8>stat.st_size:
                raise ValueError('Invalid Safetensors header size.')
            header=json.loads(f.read(length))
        spans=[]
        for name,info in header.items():
            if name=='__metadata__':continue
            if name in entries:raise ValueError('Duplicate tensor in checkpoint shards.')
            if info['dtype'] not in DTYPES:raise ValueError('Only F32, F16 and BF16 checkpoints are supported in on-demand mode.')
            shape=info['shape']
            if not isinstance(shape,list) or not 1<=len(shape)<=4 or any(not isinstance(n,int) or n<=0 for n in shape):
                raise ValueError('Invalid tensor shape.')
            count=math.prod(shape);start,end=info['data_offsets']
            if start<0 or end-start!=count*DTYPES[info['dtype']] or 8+length+end>stat.st_size:
                raise ValueError('Invalid tensor byte span.')
            spans.append((start,end))
            entries[name]=dict(file=str(file),shape=shape,source_bytes=end-start,compute_bytes=count*4,
                               fingerprint=[stat.st_size,stat.st_mtime_ns])
        spans.sort()
        if any(a[1]!=b[0] for a,b in zip(spans,spans[1:])) or (spans and spans[0][0]!=0):
            raise ValueError('Overlapping or discontinuous Safetensors payload.')
    if weight_map is not None and (set(weight_map)!=set(entries) or any(Path(entries[n]['file']).name!=v for n,v in weight_map.items())):
        raise ValueError('Checkpoint index does not match shard headers.')
    groups={}
    for name,entry in entries.items():
        module=name.rsplit('.',1)[0]
        groups[module]=groups.get(module,0)+entry['compute_bytes']
    if not groups:raise ValueError('No tensors in checkpoint.')
    return dict(path=str(root),name=root.name,architecture=arch,layers=layers,size=size,
                context_max=context,kv_per_token=2*4*layers*kv_heads*head_dim,
                tensor_count=len(entries),largest_component=max(groups.values()),
                largest_source=max(t['source_bytes'] for t in entries.values()),
                hidden_size=hidden,heads=heads,vocab_size=int(config.get('vocab_size',0)),
                tied=bool(config.get('tie_word_embeddings',False)),entries=entries,mode='streaming')

def public_model(model):
    return {k:v for k,v in model.items() if k!='entries'}

def plan_streaming(model,hw,ram_gib,vram_gib,context,profile="smart"):
    if not .5<=ram_gib<=hw['ram_total']/GIB or not 0<=vram_gib<=hw['gpu_total']/GIB:
        raise ValueError('Memory budgets exceed the detected hardware limits.')
    if not 256<=context<=min(32768,model['context_max']):
        raise ValueError('Context is outside the supported range.')
    kv=model['kv_per_token']*context
    # One leaf component at a time, plus a source view during F32 conversion.
    weights=model['largest_component']+model['largest_source']
    scratch=max(.25*GIB,4*32*(model['heads']*context+model['hidden_size']*8+model['vocab_size']))
    estimate=max(hw['app_rss'],.35*GIB)+1.0*GIB+weights+kv+scratch
    available=min(ram_gib*GIB,hw['ram_available']+hw['app_rss']-.75*GIB)
    gpu=min(vram_gib*GIB,max(0,hw['gpu_free']-.25*GIB)) if hw['gpu_ready'] else 0
    result=dict(mode='streaming',gpu_layers=0,cpu_layers=0,total_layers=model['layers'],
                disk_layers=model['layers'],ram_estimate=int(estimate),vram_estimate=0,
                ram_cap=int(ram_gib*GIB),vram_cap=int(vram_gib*GIB),context=context,kv_bytes=kv,
                placement='on-demand hybrid' if gpu>256*1024**2 else 'cpu',gpu_enabled=gpu>256*1024**2,gpu_work_budget=int(gpu),largest_component=model['largest_component'],
                fits=estimate<=available,
                reason='' if estimate<=available else 'The largest component, KV cache and runtime do not fit this RAM budget. Reduce context or use a smaller model.',
                note='No resident weight layers. Read a component from disk, execute it, then release it. KV cache stays in RAM; eligible operations can use GPU temporarily.')

    from .performance import speed_plan
    tuning=speed_plan(model,hw,result,profile)
    result.update(tuning)
    result['ram_estimate']+=tuning['performance_reserve']
    result['note']=('Automatic hybrid placement: each active component uses GPU when it fits the live budget; larger components run from RAM. GPU weights are released after use. ' if result['gpu_enabled'] else 'CPU execution: components run from RAM. ')+('Inactive reusable weights stay in a bounded RAM cache; remaining weights stay on disk.' if profile=='smart' else 'Weights are released after each operation.')
    return result
