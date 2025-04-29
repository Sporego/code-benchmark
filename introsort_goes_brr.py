# Hoare Partition QuickSort vs. Manual HeapSort vs. MergeSort Benchmark
# -----------------------------------------------------------------------
# Purpose: Compare the performance of QuickSort (Hoare partition),
#          HeapSort (manual, no heapq), MergeSort (classic recursive),
#          MergeSort (bottom-up iterative), and Introsort
#          to demonstrate why Introsort tends to be fastest in practice.

import random
import sys
import time
from typing import List, Callable

USE_NINTHERS = True  # Toggle between median-of-3 and median-of-ninthers for introsort

# ---------------------------
# Manual HeapSort Components
# ---------------------------

def _sift_down(arr: List[int], n: int, i: int) -> None:
    while True:
        left = (i << 1) + 1
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
    n = len(arr)
    for i in range((n >> 1) - 1, -1, -1):
        _sift_down(arr, n, i)
    for end in range(n - 1, 0, -1):
        arr[0], arr[end] = arr[end], arr[0]
        _sift_down(arr, end, 0)
    return arr

# --------------------------------
# QuickSort with Hoare Partition
# --------------------------------

def _hoare_partition(arr: List[int], low: int, high: int) -> int:
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
    if high is None:
        high = len(arr) - 1
    while low < high:
        p = _hoare_partition(arr, low, high)
        if p - low < high - p:
            quicksort_hoare(arr, low, p)
            low = p + 1
        else:
            quicksort_hoare(arr, p + 1, high)
            high = p
    return arr

# ---------------------------
# Pivot Helpers
# ---------------------------

def _median_of_three(arr, low, high):
    mid = (low + high) >> 1
    a, b, c = arr[low], arr[mid], arr[high]
    if a < b:
        if b < c:
            return b
        elif a < c:
            return c
        else:
            return a
    else:
        if a < c:
            return a
        elif b < c:
            return c
        else:
            return b

def _median_of_ninthers(arr, low, high):
    size = high - low + 1
    if size < 9:
        return arr[(low + high) >> 1]
    step = max(size // 9, 1)
    samples = [arr[low + i * step] for i in range(9)]
    groups = [sorted(samples[i:i+3]) for i in range(0, 9, 3)]
    medians = [g[1] for g in groups]
    return sorted(medians)[1]

# ---------------------------
# Introsort (QuickSort + HeapSort + InsertionSort + Median Pivot)
# ---------------------------

def _insertion_sort(arr: List[int], low: int, high: int) -> None:
    for i in range(low + 1, high + 1):
        temp = arr[i]
        j = i - 1
        while j >= low and arr[j] > temp:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = temp

def introsort(arr: List[int]) -> List[int]:
    maxdepth = (len(arr).bit_length() - 1) * 2
    _introsort(arr, 0, len(arr) - 1, maxdepth)
    return arr

def _introsort(arr: List[int], low: int, high: int, maxdepth: int) -> None:
    while low < high:
        if high - low + 1 <= 16:
            _insertion_sort(arr, low, high)
            return
        if maxdepth == 0:
            _heapsort_partial(arr, low, high)
            return
        pivot = _median_of_ninthers(arr, low, high) if USE_NINTHERS else _median_of_three(arr, low, high)
        i, j = low - 1, high + 1
        while True:
            i += 1
            while arr[i] < pivot:
                i += 1
            j -= 1
            while arr[j] > pivot:
                j -= 1
            if i >= j:
                break
            arr[i], arr[j] = arr[j], arr[i]
        if j - low < high - j:
            _introsort(arr, low, j, maxdepth - 1)
            low = j + 1
        else:
            _introsort(arr, j + 1, high, maxdepth - 1)
            high = j

def _heapsort_partial(arr: List[int], low: int, high: int) -> None:
    def sift(l, n, i):
        while True:
            largest = i
            left = (i << 1) + 1
            right = left + 1
            if left < n and arr[l + left] > arr[l + largest]:
                largest = left
            if right < n and arr[l + right] > arr[l + largest]:
                largest = right
            if largest == i:
                break
            arr[l + i], arr[l + largest] = arr[l + largest], arr[l + i]
            i = largest

    n = high - low + 1
    for i in range((n >> 1) - 1, -1, -1):
        sift(low, n, i)
    for end in range(n - 1, 0, -1):
        arr[low], arr[low + end] = arr[low + end], arr[low]
        sift(low, end, 0)

# ---------------------------
# MergeSort (Recursive & Iterative)
# ---------------------------

def mergesort(arr: List[int]) -> List[int]:
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = mergesort(arr[:mid])
    right = mergesort(arr[mid:])
    return _merge(left, right)

def _merge(left: List[int], right: List[int]) -> List[int]:
    merged = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged

def mergesort_bottom_up(arr: List[int]) -> List[int]:
    n = len(arr)
    width = 1
    result = arr[:]
    while width < n:
        for i in range(0, n, 2 * width):
            left = result[i:i+width]
            right = result[i+width:i+2*width]
            result[i:i+2*width] = _merge(left, right)
        width *= 2
    return result

# ---------------------------
# Benchmark Logic
# ---------------------------

def _random_array(n: int) -> List[int]:
    return [random.randint(0, n * 10) for _ in range(n)]

def _bench(fn: Callable[[List[int]], List[int]], sample: List[int], runs: int) -> float:
    total = 0.0
    for _ in range(runs):
        data = sample[:]
        start = time.perf_counter()
        fn(data)
        total += time.perf_counter() - start
    return total / runs

def main() -> None:
    if len(sys.argv) >= 2:
        N = int(sys.argv[1])
    else:
        N = 100_000
    if len(sys.argv) >= 3:
        RUNS = int(sys.argv[2])
    else:
        RUNS = 5

    print(f"Benchmarking on {N:,} integers × {RUNS} runs (Python {sys.version.split()[0]})\n")
    baseline = _random_array(N)

    hs_time = _bench(heapsort, baseline, RUNS)
    qs_time = _bench(quicksort_hoare, baseline, RUNS)
    ms_time = _bench(mergesort, baseline, RUNS)
    bu_ms_time = _bench(mergesort_bottom_up, baseline, RUNS)
    introsort_time = _bench(introsort, baseline, RUNS)

    print(f"HeapSort             : {hs_time*1e3:8.2f} ms (avg)")
    print(f"QuickSort-Hoare      : {qs_time*1e3:8.2f} ms (avg)")
    print(f"MergeSort (Recursive): {ms_time*1e3:8.2f} ms (avg)")
    print(f"MergeSort (Bottom-Up): {bu_ms_time*1e3:8.2f} ms (avg)")
    print(f"Introsort            : {introsort_time*1e3:8.2f} ms (avg)")

    if qs_time:
        print(f"→ QuickSort is {hs_time/qs_time:5.2f}× faster than HeapSort")
        print(f"→ QuickSort is {ms_time/qs_time:5.2f}× faster than Recursive MergeSort")
        print(f"→ QuickSort is {bu_ms_time/qs_time:5.2f}× faster than Bottom-Up MergeSort")
        print(f"→ QuickSort is {introsort_time/qs_time:5.2f}× faster than Introsort\n")

    a1 = baseline[:]
    a2 = baseline[:]
    a3 = baseline[:]
    a4 = baseline[:]
    a5 = baseline[:]
    assert heapsort(a1) == quicksort_hoare(a2) == mergesort(a3) == mergesort_bottom_up(a4) == introsort(a5), "Mismatch in sorting!"

if __name__ == "__main__":
    sys.setrecursionlimit(1 << 20)
    main()
