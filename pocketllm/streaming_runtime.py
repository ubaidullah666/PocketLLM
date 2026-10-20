"""Actual per-component weight paging. No whole checkpoint state_dict is loaded.

Imports are isolated from the desktop process: only the streaming worker needs
PyTorch/Transformers. Supported model classes use their parameters through leaf
module forward calls; unexpected parameter-owning parents are rejected.
"""
import gc
import time
from pathlib import Path
import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM
from .hardware import snapshot
from .streaming_model import inspect_streaming
from .performance import WeightCache
from .planner import component_fits_gpu

class Cancelled(Exception):
    pass

class ComponentPager:
    def __init__(self, model, info, allocation, cancel, emit=lambda event:None):
        self.model,self.info,self.allocation=model,info,allocation
        self.cancel,self.emit=cancel,emit
        self.resident_bytes=0;self.peak_bytes=0;self.loads=0;self.releases=0
        self.metrics={};self.disk_bytes=0;self.active=None;self.handles=[];self.last_event=0
        self.cache=WeightCache(allocation.get('weight_cache_budget',0))
        self.gpu_available=False;self.gpu_fallbacks=0
        self.gpu_component_loads=0;self.cpu_component_loads=0
        if allocation.get('gpu_enabled'):
            try:
                self.gpu_available=torch.cuda.is_available()
            except Exception:
                pass
        self.gpu_baseline=torch.cuda.memory_allocated() if self.gpu_available else 0
        modules=list(model.named_modules())
        # Validate every weight and shape before reading any payload.
        self.specs={}
        for name,module in modules:
            params=list(module.named_parameters(recurse=False))
            if not params:continue
            if any(list(c.parameters()) for c in module.children()):
                raise ValueError('This model accesses nested weights in a way the pager does not support.')
            spec=[]
            for local,param in params:
                source=f'{name}.{local}' if name else local
                if source not in info['entries'] and source=='lm_head.weight' and info['tied']:
                    source='model.embed_tokens.weight'
                entry=info['entries'].get(source)
                if not entry or list(param.shape)!=entry['shape']:
                    raise ValueError(f'Missing or incompatible checkpoint tensor: {source}')
                spec.append((local,source,tuple(param.shape),entry))
            self.specs[name]=spec
        for name,module in modules:
            if name in self.specs:
                self.handles.append(module.register_forward_pre_hook(self.make_load(name)))
                self.handles.append(module.register_forward_hook(self.make_release(name),always_call=True))
        self.report(force=True)

    def report(self, force=False):
        now=time.monotonic()
        if force or now-self.last_event>.15:
            self.last_event=now
            self.emit(dict(active=self.active,resident_weight_bytes=self.resident_bytes,
                           peak_weight_bytes=self.peak_bytes,component_loads=self.loads,
                           component_releases=self.releases,logical_weight_bytes_read=self.disk_bytes,
                           gpu_available=self.gpu_available,gpu_fallbacks=self.gpu_fallbacks,
                           gpu_component_loads=self.gpu_component_loads,cpu_component_loads=self.cpu_component_loads,
                           cached_weight_bytes=self.cache.bytes,cache_hits=self.cache.hits,
                           cache_misses=self.cache.misses,cache_evictions=self.cache.evictions,**self.metrics))

    def clear(self,name,module):
        for local,source,shape,entry in self.specs[name]:
            setattr(module,local,torch.nn.Parameter(torch.empty(shape,device='meta',dtype=torch.float32),requires_grad=False))
        self.resident_bytes=0
        self.active=None
        if self.gpu_available and not self.allocation.get("weight_cache_budget",0):
            torch.cuda.empty_cache()

    def begin_request(self):
        for file,fingerprint in {e["file"]:e["fingerprint"] for e in self.info["entries"].values()}.items():
            stat=Path(file).stat()
            if [stat.st_size,stat.st_mtime_ns]!=fingerprint:
                self.cache.trim()
                raise ValueError("Model file changed. Reload the model.")

    def make_load(self,name):
        def load(module,args):
            if self.cancel.is_set():raise Cancelled('Generation stopped.')
            if self.active is not None:raise RuntimeError('Overlapping component execution is unsupported.')
            spec=self.specs[name]
            bytes_needed=sum(e['compute_bytes'] for _,_,_,e in spec)
            source_peak=max(e['source_bytes'] for _,_,_,e in spec)
            h=snapshot()
            if h['app_rss']+bytes_needed+source_peak+256*1024**2 > self.allocation['ram_cap'] or h['ram_available']<bytes_needed+source_peak+512*1024**2:
                self.cache.trim()
                h=snapshot()
            if h['app_rss']+bytes_needed+source_peak > self.allocation['ram_cap'] or h['ram_available']<bytes_needed+source_peak+256*1024**2:
                raise MemoryError('Not enough RAM for the next component. No additional weights were loaded.')
            device='cpu'
            # Do not place a component on CUDA if an input is not a floating tensor
            # (embeddings remain CPU), or if current headroom is inadequate.
            eligible=args and all(isinstance(a,torch.Tensor) and a.is_floating_point() for a in args)
            if eligible and self.gpu_available:
                free,_=torch.cuda.mem_get_info()
                used=max(0,torch.cuda.memory_allocated()-self.gpu_baseline)
                input_bytes=sum(a.numel()*a.element_size() for a in args)
                if component_fits_gpu(bytes_needed,input_bytes,used,free,self.allocation['vram_cap']):
                    device='cuda:0'
                else:
                    self.gpu_fallbacks+=1
            self.active={'name':name,'device':device,'bytes':bytes_needed}
            try:
                for local,source,shape,entry in spec:
                    stat=Path(entry['file']).stat()
                    if [stat.st_size,stat.st_mtime_ns]!=entry['fingerprint']:
                        raise ValueError('Model file changed during inference. Reload the model.')
                    cached=self.cache.get(source)
                    if cached is not None:
                        weight=cached if device=='cpu' else cached.to(device)
                    else:
                        with safe_open(entry['file'],framework='pt',device='cpu') as f:
                            source_view=f.get_tensor(source)
                            weight=source_view.to(device=device,dtype=torch.float32,copy=True)
                            del source_view
                        self.disk_bytes+=entry['source_bytes']
                        if device=='cpu':self.cache.admit(source,weight,entry['compute_bytes'])
                    setattr(module,local,torch.nn.Parameter(weight,requires_grad=False))
                    del weight,cached
                    self.resident_bytes+=entry['compute_bytes']
                self.loads+=1
                if device=='cpu':self.cpu_component_loads+=1
                else:self.gpu_component_loads+=1
                self.peak_bytes=max(self.peak_bytes,self.resident_bytes)
                self.report()
                return tuple(a.to(device) for a in args) if device!='cpu' else args
            except Exception:
                self.clear(name,module)
                self.report(force=True)
                raise
        return load

    def make_release(self,name):
        def release(module,args,output):
            # CPU activations and KV cache outlive the temporary weight component.
            try:
                if isinstance(output,torch.Tensor) and output.device.type!='cpu':
                    output=output.to('cpu')
            finally:
                was_loaded=self.active is not None
                self.clear(name,module)
                if was_loaded:self.releases+=1
                self.report()
            return output
        return release

    def release_all(self,keep_cache=False):
        for name,module in self.model.named_modules():
            if name in self.specs:self.clear(name,module)
        if not keep_cache:
            self.cache.trim()
            gc.collect()
        if self.gpu_available:torch.cuda.empty_cache()
        self.report(force=True)


def load_skeleton(path,allocation,cancel,emit=lambda event:None):
    info=inspect_streaming(path)
    config=AutoConfig.from_pretrained(info['path'],local_files_only=True,trust_remote_code=False)
    config._attn_implementation='sdpa'
    with init_empty_weights(include_buffers=False):
        model=AutoModelForCausalLM.from_config(config,trust_remote_code=False,dtype=torch.float32)
    # No .to() on the skeleton: all weight parameters must stay on meta until used.
    model.eval()
    pager=ComponentPager(model,info,allocation,cancel,emit)
    return model,pager,info
