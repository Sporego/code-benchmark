# Hoare Partition QuickSort vs. Manual HeapSort Benchmark
# -------------------------------------------------------
# Purpose: Demonstrate that QuickSort implemented with Hoare partitioning
#          tends to outperform a classic in‑place HeapSort on large random
#          arrays, partly because the mostly sequential memory access of
#          Hoare QuickSort plays nicer with modern CPU cache‑prefetchers.
#
# Constraints: *No* use of Python's built‑in `heapq` or `sorted` helpers –
#              all data‑structure logic is written from scratch.
#
# Usage (CLI):
#     python hoare_vs_heap_benchmark.py 200000 5
# …will benchmark both algorithms on 200 000 integers for 5 trials each.

import random
import sys
import time
from typing import List, Callable

# ---------------------------
# Manual HeapSort Components
# ---------------------------

def _sift_down(arr: List[int], n: int, i: int) -> None:
    """Restore max‑heap invariant for subtree rooted at *i* (iterative)."""
    while True:
        left = (i << 1) + 1  # 2*i + 1
        right = left + 1
        largest = i

        if left < n and arr[left] > arr[largest]:
            largest = left
        if right < n and arr[right] > arr[largest]:
            largest = right

        if largest == i:
            return

        arr[i], arr[largest] = arr[largest], arr[i]
        i = largest

def heapsort(arr: List[int]) -> List[int]:
    """In‑place max‑heap sort (O(n log n), no extra memory)."""
    n = len(arr)
    # Build max‑heap bottom‑up (O(n))
    for i in range((n >> 1) - 1, -1, -1):
        _sift_down(arr, n, i)

    # Pop max element one by one
    for end in range(n - 1, 0, -1):
        arr[0], arr[end] = arr[end], arr[0]  # move current max to tail
        _sift_down(arr, end, 0)  # restore heap for the shortened prefix
    return arr

# --------------------------------
# QuickSort with Hoare Partition
# --------------------------------

def _hoare_partition(arr: List[int], low: int, high: int) -> int:
    """Classic Hoare partition (returns final index of partition point)."""
    pivot = arr[(low + high) >> 1]
    i, j = low - 1, high + 1
    while True:
        i += 1
        while arr[i] < pivot:
            i += 1
        j -= 1
        while arr[j] > pivot:
            j -= 1
        if i >= j:
            return j
        arr[i], arr[j] = arr[j], arr[i]

def quicksort_hoare(arr: List[int], low: int = 0, high: int | None = None) -> List[int]:
    """In‑place recursive QuickSort using Hoare partition scheme."""
    if high is None:
        high = len(arr) - 1
    while low < high:
        p = _hoare_partition(arr, low, high)
        # Tail recursion elimination: sort smaller side first to keep stack O(log n)
        if p - low < high - p:
            quicksort_hoare(arr, low, p)
            low = p + 1
        else:
            quicksort_hoare(arr, p + 1, high)
            high = p
    return arr

# ---------------
# Benchmark Logic
# ---------------

def _random_array(n: int) -> List[int]:
    return [random.randint(0, n * 10) for _ in range(n)]

def _bench(fn: Callable[[List[int]], List[int]], sample: List[int], runs: int) -> float:
    total = 0.0
    for _ in range(runs):
        data = sample[:]  # copy to avoid in‑place effects
        start = time.perf_counter()
        fn(data)
        total += time.perf_counter() - start
    return total / runs

def main() -> None:
    if len(sys.argv) >= 2:
        N = int(sys.argv[1])
    else:
        N = 100_000  # default size
    if len(sys.argv) >= 3:
        RUNS = int(sys.argv[2])
    else:
        RUNS = 5  # default repetitions

    print(f"Benchmarking on {N:,} integers × {RUNS} runs (Python {sys.version.split()[0]})\n")
    baseline = _random_array(N)

    hs_time = _bench(heapsort, baseline, RUNS)
    qs_time = _bench(quicksort_hoare, baseline, RUNS)

    print(f"HeapSort          : {hs_time*1e3:8.2f} ms (avg)")
    print(f"QuickSort‑Hoare   : {qs_time*1e3:8.2f} ms (avg)")
    if qs_time:
        print(f"→ QuickSort is {hs_time/qs_time:5.2f}× faster\n")

    # Small sanity check (optional):
    arr1, arr2 = baseline[:], baseline[:]
    assert heapsort(arr1) == quicksort_hoare(arr2), "Algorithms disagree on sorted order!"

if __name__ == "__main__":
    # Improve recursion head‑room for very large inputs
    sys.setrecursionlimit(1 << 20)
    main()
