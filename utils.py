import os
import multiprocessing
import platform
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial

def get_num_workers():
    cpu_count = multiprocessing.cpu_count()
    if platform.system() == 'Windows':
        return min(2, cpu_count)
    else:
        return min(4, cpu_count)

def get_dataloader_kwargs(batch_size, shuffle=True, pin_memory=None):
    import torch
    
    num_workers = get_num_workers()
    
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    
    kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
    }
    
    if platform.system() == 'Windows' and num_workers > 0:
        kwargs['multiprocessing_context'] = multiprocessing.get_context('spawn')
    
    if num_workers > 0:
        kwargs['persistent_workers'] = True
        kwargs['prefetch_factor'] = 2
    
    return kwargs

def parallel_map(func, items, max_workers=None, use_threads=True):
    if max_workers is None:
        max_workers = min(4, multiprocessing.cpu_count())
    
    executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    
    with executor_class(max_workers=max_workers) as executor:
        results = list(executor.map(func, items))
    
    return results

def batch_process(func, items, batch_size=32, max_workers=None):
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        batch_results = parallel_map(func, batch, max_workers=max_workers)
        results.extend(batch_results)
    return results

