import matplotlib.pyplot as plt
import time

def linear_search(arr, target, visualize=False):
    """
    Linear Search algorithm.
    Concept: Sequential search through the array.
    Time complexity: O(n) worst and average case.
    Space complexity: O(1).
    ML/DL applications: Simple search in unsorted data, baseline for comparison.
    Example: Finding an element in a list of predictions.
    Edge cases: Target not in array, empty array, target at start/end.
    """
    for i in range(len(arr)):
        if visualize:
            visualize_search(arr, i, target, "Linear Search")
        if arr[i] == target:
            return i
    return -1

def binary_search_iterative(arr, target, visualize=False):
    """
    Binary Search (Iterative) algorithm.
    Concept: Divide and conquer on sorted array.
    Time complexity: O(log n).
    Space complexity: O(1).
    ML/DL applications: Hyperparameter tuning (e.g., finding optimal threshold),
    searching in sorted feature lists.
    Stable: N/A for search.
    Example: Searching for a specific accuracy value in sorted results.
    Edge cases: Target not in array, empty array, single element, duplicates.
    """
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if visualize:
            visualize_search(arr, mid, target, "Binary Search Iterative", left, right)
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

def binary_search_recursive(arr, target, left=0, right=None, visualize=False):
    """
    Binary Search (Recursive) algorithm.
    Same as iterative but recursive.
    Time complexity: O(log n).
    Space complexity: O(log n) due to recursion.
    ML applications: Same as iterative, but recursion depth limit in Python.
    """
    if right is None:
        right = len(arr) - 1
    if left > right:
        return -1
    mid = (left + right) // 2
    if visualize:
        visualize_search(arr, mid, target, "Binary Search Recursive", left, right)
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right, visualize)
    else:
        return binary_search_recursive(arr, target, left, mid - 1, visualize)

def interpolation_search(arr, target, visualize=False):
    """
    Interpolation Search algorithm.
    Concept: Improved binary search for uniformly distributed data,
    probes position based on value.
    Time complexity: O(log log n) average for uniform, O(n) worst.
    Space complexity: O(1).
    ML/DL applications: Searching in sorted arrays with uniform distribution,
    e.g., searching timestamps or scores.
    Example: Finding a value in a sorted list of evenly spaced numbers.
    Edge cases: Non-uniform data (falls back to linear), target out of range.
    """
    low, high = 0, len(arr) - 1
    while low <= high and target >= arr[low] and target <= arr[high]:
        if low == high:
            if arr[low] == target:
                return low
            return -1
        pos = low + int(((float(high - low) / (arr[high] - arr[low])) * (target - arr[low])))
        if visualize:
            visualize_search(arr, pos, target, "Interpolation Search", low, high)
        if arr[pos] == target:
            return pos
        elif arr[pos] < target:
            low = pos + 1
        else:
            high = pos - 1
    return -1

def visualize_search(arr, current_idx, target, title, left=None, right=None, pause=0.5):
    """
    Visualize the search process.
    Highlights current index, and for binary, left and right.
    """
    plt.clf()
    colors = ['skyblue'] * len(arr)
    if current_idx is not None and 0 <= current_idx < len(arr):
        colors[current_idx] = 'red'  # Current probe
    if left is not None and right is not None:
        for i in range(left, right + 1):
            if colors[i] == 'skyblue':
                colors[i] = 'lightgreen'  # Search range
    plt.bar(range(len(arr)), arr, color=colors)
    plt.title(f"{title} - Target: {target}, Current: {current_idx}")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.pause(pause)
    plt.draw()

def visualize_search_process(search_func, arr, target, title, *args, **kwargs):
    """
    Visualize the entire search process.
    """
    plt.ion()
    result = search_func(arr, target, visualize=True, *args, **kwargs)
    plt.ioff()
    plt.show()
    return result

if __name__ == "__main__":
    # Sorted array for binary and interpolation
    arr = [2, 3, 4, 10, 40, 50, 60, 70, 80, 90]
    target = 10

    print("Linear Search:")
    idx = linear_search(arr.copy(), target)
    print(f"Found at index: {idx}")

    print("Binary Search Iterative:")
    idx = binary_search_iterative(arr, target)
    print(f"Found at index: {idx}")

    print("Binary Search Recursive:")
    idx = binary_search_recursive(arr, target)
    print(f"Found at index: {idx}")

    print("Interpolation Search:")
    idx = interpolation_search(arr, target)
    print(f"Found at index: {idx}")

    # Visualizations
    visualize_search_process(linear_search, arr.copy(), target, "Linear Search")
    visualize_search_process(binary_search_iterative, arr, target, "Binary Search Iterative")
    visualize_search_process(binary_search_recursive, arr, target, "Binary Search Recursive")
    visualize_search_process(interpolation_search, arr, target, "Interpolation Search")