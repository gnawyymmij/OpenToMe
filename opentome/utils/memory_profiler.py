"""
Memory profiling utilities for debugging GPU memory usage
"""
import torch
from collections import defaultdict
import time


class MemoryProfiler:
    """
    Context manager and decorator for profiling GPU memory usage
    """
    def __init__(self, name="", enabled=True, verbose=True):
        self.name = name
        self.enabled = enabled
        self.verbose = verbose
        self.start_alloc = 0
        self.start_reserved = 0
        self.start_time = 0
        
    def __enter__(self):
        if self.enabled and torch.cuda.is_available():
            torch.cuda.synchronize()
            self.start_alloc = torch.cuda.memory_allocated()
            self.start_reserved = torch.cuda.memory_reserved()
            self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        if self.enabled and torch.cuda.is_available():
            torch.cuda.synchronize()
            end_alloc = torch.cuda.memory_allocated()
            end_reserved = torch.cuda.memory_reserved()
            elapsed = time.time() - self.start_time
            
            delta_alloc = (end_alloc - self.start_alloc) / 1024**2
            delta_reserved = (end_reserved - self.start_reserved) / 1024**2
            current_alloc = end_alloc / 1024**2
            current_reserved = end_reserved / 1024**2
            
            if self.verbose:
                print(f"[MEM] {self.name:30s} | "
                      f"Δ Alloc: {delta_alloc:+7.1f}MB | "
                      f"Δ Reserved: {delta_reserved:+7.1f}MB | "
                      f"Current: {current_alloc:7.1f}MB / {current_reserved:7.1f}MB | "
                      f"Time: {elapsed*1000:.1f}ms", flush=True)


class DetailedMemoryTracker:
    """
    Track memory usage across multiple calls with statistics
    """
    def __init__(self):
        self.records = defaultdict(list)
        
    def record(self, name, mem_delta, time_delta=None):
        self.records[name].append({
            'mem_delta': mem_delta,
            'time_delta': time_delta
        })
    
    def report(self, top_k=10):
        print("\n" + "="*80)
        print("Memory Usage Summary (Top consumers)")
        print("="*80)
        
        # Sort by total memory
        stats = []
        for name, records in self.records.items():
            mem_deltas = [r['mem_delta'] for r in records]
            time_deltas = [r['time_delta'] for r in records if r['time_delta'] is not None]
            
            stats.append({
                'name': name,
                'total_mem': sum(mem_deltas),
                'avg_mem': sum(mem_deltas) / len(mem_deltas),
                'max_mem': max(mem_deltas),
                'count': len(mem_deltas),
                'avg_time': sum(time_deltas) / len(time_deltas) if time_deltas else 0
            })
        
        stats.sort(key=lambda x: x['total_mem'], reverse=True)
        
        print(f"{'Module':<30s} | {'Total':>10s} | {'Avg':>10s} | {'Max':>10s} | {'Count':>6s} | {'Avg Time':>10s}")
        print("-"*80)
        
        for stat in stats[:top_k]:
            print(f"{stat['name']:<30s} | "
                  f"{stat['total_mem']:>9.1f}M | "
                  f"{stat['avg_mem']:>9.1f}M | "
                  f"{stat['max_mem']:>9.1f}M | "
                  f"{stat['count']:>6d} | "
                  f"{stat['avg_time']*1000:>9.1f}ms")
        
        print("="*80 + "\n")


def profile_module_forward(module, name, profiler=None):
    """
    Wrap a module's forward method with memory profiling
    """
    original_forward = module.forward
    
    def profiled_forward(*args, **kwargs):
        with MemoryProfiler(name, enabled=(profiler is not None), verbose=False) as prof:
            result = original_forward(*args, **kwargs)
        
        if profiler is not None:
            delta = (torch.cuda.memory_allocated() - prof.start_alloc) / 1024**2
            time_delta = time.time() - prof.start_time
            profiler.record(name, delta, time_delta)
        
        return result
    
    module.forward = profiled_forward
    return module


def get_tensor_memory(tensor):
    """Get memory usage of a tensor in MB"""
    if tensor is None:
        return 0
    if isinstance(tensor, (list, tuple)):
        return sum(get_tensor_memory(t) for t in tensor)
    if not isinstance(tensor, torch.Tensor):
        return 0
    return tensor.element_size() * tensor.nelement() / 1024**2


def print_tensor_shapes(name, *tensors):
    """Print shapes and memory of tensors for debugging"""
    print(f"\n[TENSOR] {name}:")
    for i, t in enumerate(tensors):
        if isinstance(t, torch.Tensor):
            mem = get_tensor_memory(t)
            print(f"  [{i}] shape={tuple(t.shape)}, dtype={t.dtype}, mem={mem:.1f}MB")
        elif isinstance(t, (list, tuple)):
            total_mem = sum(get_tensor_memory(x) for x in t if isinstance(x, torch.Tensor))
            print(f"  [{i}] list/tuple of {len(t)} items, total_mem={total_mem:.1f}MB")
        elif t is None:
            print(f"  [{i}] None")
        else:
            print(f"  [{i}] {type(t)}")

