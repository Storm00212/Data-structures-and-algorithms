import matplotlib.pyplot as plt
import time
from Trees_and_Heaps.heap_implementations import heap_sort

def partition(arr, low, high, visualize=False):
    """
    Partition function for Quick Sort.
    Chooses the last element as pivot, places it in correct position,
    and partitions the array around it.
    """
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
            if visualize:
                visualize_array(arr, f"Quick Sort: Swapping {arr[i]} and {arr[j]}")
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    if visualize:
        visualize_array(arr, f"Quick Sort: Placing pivot {pivot}")
    return i + 1

def quick_sort(arr, low=0, high=None, visualize=False):
    """
    Quick Sort algorithm.
    Concept: Divide and conquer, recursive partitioning.
    Time complexity: O(n log n) average, O(n^2) worst case.
    Space complexity: O(log n) due to recursion stack.
    ML/DL implications: Efficient for large datasets in preprocessing,
    but worst case can be avoided with good pivot selection (e.g., median of three).
    Stable: No.
    Example: Sorting feature values in ML datasets.
    Edge cases: Already sorted (worst case), reverse sorted, duplicates.
    """
    if high is None:
        high = len(arr) - 1
    if low < high:
        pi = partition(arr, low, high, visualize)
        quick_sort(arr, low, pi - 1, visualize)
        quick_sort(arr, pi + 1, high, visualize)

def merge(arr, left, mid, right, visualize=False):
    """
    Merge function for Merge Sort.
    Merges two sorted subarrays into one.
    """
    n1 = mid - left + 1
    n2 = right - mid
    L = arr[left:left + n1]
    R = arr[mid + 1:mid + 1 + n2]
    i = j = 0
    k = left
    while i < n1 and j < n2:
        if L[i] <= R[j]:
            arr[k] = L[i]
            i += 1
        else:
            arr[k] = R[j]
            j += 1
        k += 1
        if visualize:
            visualize_array(arr, f"Merge Sort: Merging at index {k}")
    while i < n1:
        arr[k] = L[i]
        i += 1
        k += 1
    while j < n2:
        arr[k] = R[j]
        j += 1
        k += 1

def merge_sort(arr, left=0, right=None, visualize=False):
    """
    Merge Sort algorithm.
    Concept: Divide and conquer, recursive splitting and merging.
    Time complexity: O(n log n) worst, best, average.
    Space complexity: O(n) for auxiliary arrays.
    ML/DL implications: Stable sort, good for maintaining order in features,
    used in external sorting for large datasets.
    Stable: Yes.
    Example: Sorting time-series data in DL.
    Edge cases: Empty array, single element, already sorted.
    """
    if right is None:
        right = len(arr) - 1
    if left < right:
        mid = (left + right) // 2
        merge_sort(arr, left, mid, visualize)
        merge_sort(arr, mid + 1, right, visualize)
        merge(arr, left, mid, right, visualize)

def visualize_array(arr, title="Array", pause=0.5):
    """
    Visualize the array as a bar chart.
    """
    plt.clf()
    plt.bar(range(len(arr)), arr, color='skyblue')
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.pause(pause)
    plt.draw()

def visualize_sorting_process(sort_func, arr, title, *args, **kwargs):
    """
    Visualize the sorting process.
    """
    plt.ion()
    visualize_array(arr, f"{title} - Initial")
    sort_func(arr, *args, visualize=True, **kwargs)
    visualize_array(arr, f"{title} - Sorted")
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    # Example arrays
    arr1 = [10, 7, 8, 9, 1, 5]
    arr2 = [12, 11, 13, 5, 6, 7]
    arr3 = [64, 34, 25, 12, 22, 11, 90]

    print("Quick Sort:")
    quick_sort(arr1.copy())
    print("Sorted:", arr1)

    print("Merge Sort:")
    merge_sort(arr2.copy())
    print("Sorted:", arr2)

    print("Heap Sort:")
    sorted_arr = heap_sort(arr3.copy())
    print("Sorted:", sorted_arr)

    # Visualizations
    visualize_sorting_process(quick_sort, arr1.copy(), "Quick Sort")
    visualize_sorting_process(merge_sort, arr2.copy(), "Merge Sort")
    # For heap sort, since it's not in-place, visualize differently
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.bar(range(len(arr3)), arr3, color='lightcoral')
    plt.title("Original Array")
    plt.subplot(1, 2, 2)
    sorted_arr = heap_sort(arr3.copy())
    plt.bar(range(len(sorted_arr)), sorted_arr, color='lightgreen')
    plt.title("Sorted Array (Heap Sort)")
    plt.show()