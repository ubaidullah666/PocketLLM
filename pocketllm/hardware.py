import os
import platform
import psutil

try:
    import pynvml as nv
    nv.nvmlInit()
except Exception:
    nv = None

def app_memory():
    try:
        root = psutil.Process()
    except psutil.NoSuchProcess:
        # Some container runtimes expose a host PID namespace through /proc.
        with open('/proc/self/stat') as f:
            root = psutil.Process(int(f.read().split()[0]))
    total = 0
    for p in [root]+root.children(recursive=True):
        try:
            total += p.memory_info().rss
        except (psutil.NoSuchProcess,psutil.AccessDenied):
            pass
    return total

def resolve_engine(executable=''):
    import shutil
    from pathlib import Path
    bundled = Path(__file__).resolve().parent.parent/'tools'/'llama-vulkan'/('llama-server.exe' if os.name == 'nt' else 'llama-server')
    if not bundled.is_file():
        bundled = bundled.parent.parent/'llama.cpp'/bundled.name
    return shutil.which(str(Path(executable).expanduser()) if executable else (str(bundled) if bundled.is_file() else 'llama-server'))


def engine_devices(executable=''):
    import subprocess
    import re
    resolved = resolve_engine(executable)
    if not resolved:
        return []
    env = {k:v for k,v in os.environ.items() if not k.startswith('LLAMA_ARG_')}
    try:
        output = subprocess.run([resolved,'--list-devices'], stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, errors='replace', timeout=15, env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    devices = []
    for device, name, total, free in re.findall(r'^\s*([\w]+\d+):\s*(.*?)\s*\((\d+) MiB, (\d+) MiB free\)', output, re.M):
        integrated = bool(re.search(r'Intel.*(?:UHD|HD Graphics|Iris)|Radeon.*(?:Graphics|[678]\d\dM)',name,re.I)) or bool(re.search(r'ggml_vulkan:.*?'+re.escape(name.split(' (')[0])+r'.*?uma: 1',output))
        devices.append(dict(id=device,name=name,kind='integrated / shared' if integrated else 'dedicated',
            backend=re.sub(r'\d+$','',device),total=int(total)*1024**2,free=int(free)*1024**2,
            used=(int(total)-int(free))*1024**2,uuid='',ready=True,telemetry=False))
    return devices


def display_adapters():
    """Windows display inventory; registry sizes are never used as allocation quotas."""
    if os.name != 'nt': return []
    import winreg
    adapters=[]
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}') as root:
            for i in range(winreg.QueryInfoKey(root)[0]):
                key_name=winreg.EnumKey(root,i)
                if not key_name.isdigit(): continue
                try:
                    with winreg.OpenKey(root,key_name) as key:
                        name=winreg.QueryValueEx(key,'DriverDesc')[0]
                        adapters.append(dict(id='display:'+key_name,name=name,kind='display adapter',backend='unavailable',
                            total=0,free=0,used=0,uuid='',ready=False,telemetry=False))
                except OSError: pass
    except OSError: pass
    return adapters


def snapshot(gpu_id=None, executable='', mode='resident', devices=None):
    mem = psutil.virtual_memory()
    gpus = []
    if nv:
        try:
            for index in range(nv.nvmlDeviceGetCount()):
                handle = nv.nvmlDeviceGetHandleByIndex(index)
                gm = nv.nvmlDeviceGetMemoryInfo(handle)
                name, uuid = nv.nvmlDeviceGetName(handle), nv.nvmlDeviceGetUUID(handle)
                gpus.append(dict(id='CUDA'+str(index),name=name.decode() if isinstance(name,bytes) else name,
                    uuid=uuid.decode() if isinstance(uuid,bytes) else uuid,kind='dedicated',backend='CUDA',
                    total=gm.total,free=gm.free,used=gm.used,ready=True,telemetry=True))
        except Exception:
            pass
    if devices is not None:
        for device in devices:
            if not any(g['id']==device['id'] for g in gpus):
                gpus.append(device.copy())
    for adapter in display_adapters():
        if not any(adapter['name'].casefold() in g['name'].casefold() for g in gpus):
            gpus.append(adapter)
    selected = next((g for g in gpus if g['id']==gpu_id),None) if gpu_id else next((g for g in gpus if g['ready']),None)
    if gpu_id and gpu_id!='none' and selected is None:
        raise ValueError('Selected GPU is unavailable. Refresh devices and select it again.')
    if gpu_id=='none': selected=None
    result = dict(ram_total=mem.total,ram_available=mem.available,ram_used=mem.total-mem.available,
        app_rss=app_memory(),cpu_percent=psutil.cpu_percent(),cpu_name=platform.processor() or platform.machine(),
        gpus=gpus,gpu_id=selected['id'] if selected else 'none',gpu_name='CPU mode',gpu_total=0,
        gpu_free=0,gpu_used=0,gpu_uuid='',gpu_ready=False,gpu_telemetry=False,
        gpu_note='Select a GPU-capable llama.cpp engine for dedicated or integrated GPU offload.')
    if selected:
        ready = selected['ready'] and (mode!='streaming' or selected['backend']=='CUDA')
        result.update(gpu_name=selected['name'],gpu_total=selected['total'],gpu_free=selected['free'],
            gpu_used=selected['used'],gpu_uuid=selected['uuid'],gpu_ready=ready,
            gpu_shared=selected['kind']=='integrated / shared',gpu_telemetry=selected['telemetry'],gpu_backend=selected['backend'],
            gpu_note=('This adapter was detected, but the engine does not expose it. Select a Vulkan GPU build for Resident mode.' if not selected['ready'] else 'On-demand mode supports NVIDIA CUDA only. Select Resident for this GPU.' if not ready else
                selected['backend']+' · '+selected['kind']+(' · engine-reported memory; live device telemetry unavailable' if not selected['telemetry'] else '')))
    return result
