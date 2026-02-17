
import os
import sys
import time
import multiprocessing
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import contextlib
import importlib

# Mocking the simulator.py environment

def available_cpus() -> int:
    return os.cpu_count() or 1

THREAD_ENV_VARS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}

try:
    from threadpoolctl import threadpool_limits as _threadpool_limits
    threadpool_limits = _threadpool_limits
except ImportError:
    threadpool_limits = None
    print("threadpoolctl not found")

def _limit_worker_threads(n_threads: int = 1) -> None:
    print(f"Worker initializing: PID={os.getpid()}")
    for k in THREAD_ENV_VARS:
        os.environ.setdefault(k, str(n_threads))
    os.environ.setdefault("OMP_DYNAMIC", "FALSE")
    os.environ.setdefault("MKL_DYNAMIC", "FALSE")

    with contextlib.suppress(Exception):
        import numexpr
        numexpr.set_num_threads(n_threads)

    with contextlib.suppress(Exception):
        import mkl
        mkl.set_num_threads(n_threads)

    if threadpool_limits is not None:
        with contextlib.suppress(Exception):
            threadpool_limits(limits=n_threads)
    print(f"Worker initialized: PID={os.getpid()}")

def backend_task(x):
    # Simulate some work
    import numpy as np
    # Trigger some linear algebra that might use OpenBLAS/MKL
    a = np.random.rand(100, 100)
    b = np.dot(a, a)
    return np.sum(b) + x

def main():
    print("Starting reproduction script...")
    max_workers = available_cpus()
    print(f"Max workers: {max_workers}")
    
    ctx = multiprocessing.get_context("spawn")
    
    tasks = range(50)
    
    print("Creating ProcessPoolExecutor...")
    try:
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=ctx,
            initializer=_limit_worker_threads,
            initargs=(1,),
        ) as ex:
            print("Executor created. Submitting tasks...")
            futures = {ex.submit(backend_task, i): i for i in tasks}
            
            print("Waiting for results...")
            for fut in tqdm(concurrent.futures.as_completed(futures), total=len(tasks)):
                try:
                    res = fut.result()
                except Exception as e:
                    print(f"Task failed with error: {e}")
                    raise
    except Exception as e:
        print(f"Main loop crashed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
