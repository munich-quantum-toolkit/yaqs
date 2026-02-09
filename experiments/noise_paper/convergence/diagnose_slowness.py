import time
import os
import multiprocessing
import numpy as np
from concurrent.futures import ProcessPoolExecutor, wait
import sys

# Try to import numba to check status
try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# Try to import threadpoolctl
try:
    import threadpoolctl
    HAS_THREADPOOLCTL = True
except ImportError:
    HAS_THREADPOOLCTL = False

def heavy_work(x):
    # Simulate work that might trigger BLAS
    # Create random matrix
    np.random.seed(x)
    A = np.random.rand(500, 500)
    B = np.random.rand(500, 500)
    # This should use BLAS threads if not limited
    C = np.dot(A, B)
    return np.linalg.norm(C)

def numba_work(x):
    if not HAS_NUMBA:
        return 0
    # A function that requires compilation
    @numba.jit(nopython=True, cache=True)
    def _inner(n):
        s = 0.0
        for i in range(n):
            s += float(i)
        return s
    return _inner(x * 1000)

def main():
    print(f"OS: {sys.platform}")
    print(f"Python: {sys.version}")
    print(f"Numba: {numba.__version__ if HAS_NUMBA else 'Not Found'}")
    print(f"threadpoolctl: {threadpoolctl.__version__ if HAS_THREADPOOLCTL else 'Not Found'}")
    
    cpus = os.cpu_count()
    print(f"Physical CPUs (os.cpu_count): {cpus}")
    
    # Check affinity
    if hasattr(os, "sched_getaffinity"):
        print(f"Affinity: {len(os.sched_getaffinity(0))}")

    # 1. Benchmark Spawn Overhead
    print("\n--- Benchmarking 'spawn' overhead ---")
    ctx = multiprocessing.get_context("spawn")
    workers = min(16, cpus or 1)
    
    start = time.time()
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futures = [ex.submit(heavy_work, i) for i in range(workers)]
        wait(futures)
    end = time.time()
    print(f"Time to run {workers} dot products (500x500) with 'spawn': {end - start:.4f}s")
    
    # 2. Benchmark Numba Compilation Lock (if applicable)
    if HAS_NUMBA:
        print("\n--- Benchmarking Numba Compilation/Cache Lock ---")
        # clear simple function cache if possible?
        # running parallel numba
        ctx = multiprocessing.get_context("spawn")
        start = time.time()
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futures = [ex.submit(numba_work, i) for i in range(workers)]
            wait(futures)
        end = time.time()
        print(f"Time to run {workers} Numba functions with 'spawn': {end - start:.4f}s")
        print("If this is very slow, Numba cache locking might be the issue.")

if __name__ == "__main__":
    main()

