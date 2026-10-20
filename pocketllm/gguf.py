"""Bounded, metadata-only GGUF v2/v3 reader. Never reads tensor payloads."""
import os
import re
import struct
from pathlib import Path

SUPPORTED = {'llama', 'qwen2', 'qwen3'}
FORMATS = {0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}

class GGUFError(ValueError):
    pass

class Reader:
    def __init__(self, f, size):
        self.f, self.size = f, size
    def read(self, n):
        if n < 0 or self.f.tell() + n > self.size:
            raise GGUFError('Truncated or invalid GGUF file.')
        data = self.f.read(n)
        if len(data) != n:
            raise GGUFError('Truncated GGUF file.')
        return data
    def num(self, fmt):
        return struct.unpack('<'+fmt, self.read(struct.calcsize('<'+fmt)))[0]
    def string(self, keep=True):
        n = self.num('Q')
        if n > 64*1024*1024:
            raise GGUFError('GGUF string exceeds parser safety limit.')
        if keep:
            return self.read(n).decode('utf-8', errors='replace')
        self.skip(n)
    def skip(self, n):
        if n < 0 or self.f.tell()+n > self.size:
            raise GGUFError('Invalid metadata length.')
        self.f.seek(n, 1)
    def value(self, kind, keep=True, depth=0):
        if kind in FORMATS:
            return self.num(FORMATS[kind])
        if kind == 8:
            return self.string(keep)
        if kind == 9:
            if depth > 1:
                raise GGUFError('Nested GGUF arrays are unsupported.')
            subtype, count = self.num('I'), self.num('Q')
            if count > 10_000_000:
                raise GGUFError('Metadata array exceeds safety limit.')
            if subtype in FORMATS:
                self.skip(count*struct.calcsize('<'+FORMATS[subtype]))
            else:
                for _ in range(count):
                    self.value(subtype, False, depth+1)
            return None
        raise GGUFError('Unknown metadata type.')

def inspect_model(path):
    p = Path(path).expanduser().resolve(strict=True)
    if not p.is_file() or p.suffix.lower() != '.gguf':
        raise GGUFError('Choose a local .gguf file.')
    size = p.stat().st_size
    with p.open('rb') as f:
        r = Reader(f, size)
        if r.read(4) != b'GGUF' or r.num('I') not in (2,3):
            raise GGUFError('Only little-endian GGUF v2/v3 is supported.')
        tensors, items = r.num('Q'), r.num('Q')
        if not 0 < tensors <= 1_000_000 or items > 100_000:
            raise GGUFError('Invalid GGUF header counts.')
        meta = {}
        for _ in range(items):
            key = r.string()
            if len(key) > 4096:
                raise GGUFError('Metadata key too long.')
            kind = r.num('I')
            value = r.value(kind, not key.startswith('tokenizer.'))
            if not key.startswith('tokenizer.'):
                meta[key] = value
        arch = meta.get('general.architecture')
        if arch not in SUPPORTED or meta.get(f'{arch}.expert_count',0):
            raise GGUFError('This version supports dense Llama, Qwen2 and Qwen3 GGUF models only; MoE/hybrid layouts need a different memory estimator.')
        if meta.get('split.count',1) != 1:
            raise GGUFError('Split GGUF files are not supported. Select a single-file model.')
        layers = int(meta.get(f'{arch}.block_count',0))
        if not 1 <= layers <= 1024:
            raise GGUFError('Missing or invalid layer count.')
        table = []
        for _ in range(tensors):
            name = r.string()
            dims = r.num('I')
            if not 1 <= dims <= 4:
                raise GGUFError('Invalid tensor dimensions.')
            for _ in range(dims):
                if r.num('Q') == 0:
                    raise GGUFError('Empty tensor.')
            r.num('I')  # GGML tensor type; storage span comes from offsets.
            table.append((r.num('Q'), name))
        alignment = int(meta.get('general.alignment',32))
        if alignment < 1 or alignment > 65536 or alignment & (alignment-1):
            raise GGUFError('Invalid tensor alignment.')
        start = (f.tell()+alignment-1)//alignment*alignment
        table.sort()
        blocks = [0]*layers
        for i,(offset,name) in enumerate(table):
            end = table[i+1][0] if i+1 < len(table) else size-start
            if offset % alignment or end <= offset or start+end > size:
                raise GGUFError('Invalid tensor offsets or truncated model.')
            match = re.match(r'blk\.(\d+)\.',name)
            if match:
                idx = int(match[1])
                if idx >= layers:
                    raise GGUFError('Tensor layer index outside declared model.')
                blocks[idx] += end-offset
        if not all(blocks):
            raise GGUFError('Unrecognized transformer block layout.')
        heads = int(meta.get(f'{arch}.attention.head_count',0))
        kv_heads = int(meta.get(f'{arch}.attention.head_count_kv',heads))
        emb = int(meta.get(f'{arch}.embedding_length',0))
        if not heads or not kv_heads or not emb or emb % heads:
            raise GGUFError('Missing attention dimensions for KV cache estimate.')
        key_dim = int(meta.get(f'{arch}.attention.key_length',emb//heads))
        val_dim = int(meta.get(f'{arch}.attention.value_length',emb//heads))
        if min(key_dim,val_dim) <= 0:
            raise GGUFError('Invalid attention dimensions.')
        return dict(path=str(p), name=p.name, architecture=arch, layers=layers,
                    size=size, block_bytes=blocks, kv_per_token=2*layers*kv_heads*(key_dim+val_dim),
                    context_max=int(meta.get(f'{arch}.context_length',2048)),
                    fingerprint=[size,p.stat().st_mtime_ns])
