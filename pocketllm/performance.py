"""Small bounded cache with admission that avoids cyclic LRU thrashing."""
from collections import OrderedDict

class WeightCache:
    def __init__(self,budget):
        self.budget=max(0,int(budget));self.bytes=0;self.items=OrderedDict()
        self.hits=0;self.misses=0;self.evictions=0
    def get(self,key):
        item=self.items.get(key)
        if item is None:
            self.misses+=1;return None
        self.hits+=1;self.items.move_to_end(key)
        return item[0]
    def admit(self,key,value,size):
        # Stable admission: do not evict useful weights on every sequential pass.
        # A cycling dense-model traversal would make ordinary LRU miss forever.
        if key in self.items or size>self.budget-self.bytes:return False
        self.items[key]=(value,size);self.bytes+=size;return True
    def trim(self,target=0):
        target=max(0,int(target))
        while self.items and self.bytes>target:
            _,(_,size)=self.items.popitem(last=False)
            self.bytes-=size;self.evictions+=1


def speed_plan(model,hw,base,profile='smart'):
    if profile not in ('smart','strict'):raise ValueError('Unknown performance profile.')
    from .planner import GIB
    available=min(base['ram_cap'],hw['ram_available']+hw['app_rss']-.75*GIB)
    spare=max(0,available-base['ram_estimate']-.25*GIB)
    # Larger prefill batches reduce full model traversals for the prompt. Always
    # charge the extra scratch memory before assigning any weight-cache budget.
    per_token=4*(model['heads']*base['context']+model['hidden_size']*8+model['vocab_size'])
    batch=32
    if profile=='smart':
        for candidate in (64,128,256):
            extra=(candidate-32)*per_token
            if extra<=spare*.4:batch=candidate
    extra=(batch-32)*per_token
    total=sum(e['compute_bytes'] for e in model.get('entries',{}).values())
    budget=int(min(max(0,spare-extra),base['ram_cap']*.3,total*.5)) if profile=='smart' else 0
    return dict(profile=profile,prefill_batch=batch,weight_cache_budget=budget,
                performance_reserve=int(extra+budget),reuse_prefix=profile=='smart')

class PrefixCache:
    """Own one KV cache. Reuse only a byte-for-byte identical token prefix."""
    def __init__(self):self.ids=[];self.state=None
    def take(self,ids):
        state,old=self.state,self.ids
        self.clear()
        if state is not None and len(old)<len(ids) and ids[:len(old)]==old:
            return state,len(old)
        return None,0
    def save(self,ids,state):self.ids=list(ids);self.state=state
    def clear(self):self.ids=[];self.state=None
