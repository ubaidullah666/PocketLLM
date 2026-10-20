"""Conservative estimates; hard instantaneous allocator quotas are not implied."""
GIB = 1024**3

def plan(model, hardware, ram_gib, vram_gib, context):
    if not 0.5 <= ram_gib <= hardware['ram_total']/GIB:
        raise ValueError('RAM budget must be between 0.5 GiB and installed RAM.')
    if not 0 <= vram_gib <= hardware['gpu_total']/GIB:
        raise ValueError('VRAM budget exceeds detected capacity.')
    if not 256 <= context <= min(32768,model['context_max']):
        raise ValueError('Context is outside this model’s supported range.')
    # Keep system headroom and account for current app RSS already in available RAM.
    ram_limit = min(ram_gib*GIB, hardware['ram_available']+hardware['app_rss']-0.75*GIB)
    gpu_limit = min(vram_gib*GIB, max(0,hardware['gpu_free']-0.25*GIB)) if hardware['gpu_ready'] else 0
    # KV stays on CPU; one inference slot, small 128-token batches, no repacking.
    kv = model['kv_per_token']*context
    cpu_overhead = max(hardware['app_rss'],0.35*GIB)+0.4*GIB+kv
    gpu_overhead = 0.3*GIB
    moved, selected = 0, 0
    for size in reversed(model['block_bytes']):
        if (moved+size)*1.12+gpu_overhead > gpu_limit:
            break
        moved += size
        selected += 1
    gpu_est = moved*1.12+gpu_overhead if selected else 0
    cpu_est = (model['size']-(0 if hardware.get('gpu_shared') else moved))*1.12+cpu_overhead+max(model['block_bytes'])
    if vram_gib and hardware.get('gpu_id') and not hardware['gpu_ready']:
        raise ValueError(hardware['gpu_note'])
    fits = cpu_est <= ram_limit and gpu_est <= gpu_limit
    fallback = bool(vram_gib and hardware['gpu_ready'] and not selected)
    placement_note = (f'Automatic hybrid placement: {selected} layers on GPU, {model["layers"]-selected} layers in RAM. '
                      if selected else 'GPU budget is too small for a complete layer; all layers will run from RAM. '
                      if fallback else 'CPU execution: all layers stay in RAM. ')
    return dict(placement='hybrid' if selected else 'cpu',gpu_fallback=fallback,gpu_layers=selected,cpu_layers=model['layers']-selected,total_layers=model['layers'],
                ram_estimate=int(cpu_est),vram_estimate=int(gpu_est),kv_bytes=kv,
                ram_cap=int(ram_gib*GIB),vram_cap=int(vram_gib*GIB),context=context,
                fits=fits,reason='' if fits else 'Model does not fit the available RAM budget. Reduce context, close other apps, or use a smaller quantized model.',
                note=placement_note+'Output/embedding tensors and KV cache remain in RAM. All layers are used for each token.')


def component_fits_gpu(weight_bytes, input_bytes, used_bytes, free_bytes, budget_bytes):
    """Keep working memory and driver headroom inside a transient GPU budget."""
    needed = weight_bytes + 256*1024**2 + 4*input_bytes
    return needed + used_bytes < budget_bytes and needed < free_bytes - 128*1024**2
