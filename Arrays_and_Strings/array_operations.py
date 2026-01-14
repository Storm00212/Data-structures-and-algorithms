import numpy as np
import matplotlib.pyplot as plt

class DynamicArray:
    """
    A simple implementation of a dynamic array to demonstrate dynamic resizing.
    In Python, lists are inherently dynamic, but this class simulates the concept.

    Concept: Dynamic arrays automatically resize when capacity is exceeded, typically doubling in size.
    Time Complexity: Amortized O(1) for append operations due to occasional O(n) resizing.
    Space Complexity: O(n), where n is the number of elements.
    ML/DL Implications: Useful for handling variable-sized datasets, such as growing feature vectors during preprocessing.
    Example Use Case: Preprocessing numerical data where the size isn't known in advance.
    Edge Cases: Empty array (capacity starts at 1), single element.
    """
    def __init__(self):
        self.array = [0] * 1  # Start with capacity 1
        self.size = 0
        self.capacity = 1

    def append(self, value):
        if self.size == self.capacity:
            self._resize(self.capacity * 2)
        self.array[self.size] = value
        self.size += 1

    def _resize(self, new_capacity):
        new_array = [0] * new_capacity
        for i in range(self.size):
            new_array[i] = self.array[i]
        self.array = new_array
        self.capacity = new_capacity

    def get_array(self):
        return self.array[:self.size]

def binary_search(arr, target):
    """
    Binary search on a sorted array.

    Concept: Divide and conquer approach to find an element in a sorted list.
    Time Complexity: O(log n), where n is the length of the array.
    Space Complexity: O(1), iterative implementation.
    ML/DL Implications: Efficient for feature selection in sorted datasets, e.g., finding optimal thresholds in decision trees.
    Example Use Case: Searching for a specific value in a sorted list of numerical features.
    Edge Cases: Empty array (return -1), single element (check if matches), target not found.
    """
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

def remove_duplicates(arr):
    """
    Remove duplicates from a sorted array using two-pointer technique.

    Concept: Use two pointers to track unique elements.
    Time Complexity: O(n), where n is the length of the array.
    Space Complexity: O(1), modifies in-place.
    ML/DL Implications: Data preprocessing to reduce redundancy in feature sets.
    Example Use Case: Cleaning duplicate entries in a dataset.
    Edge Cases: Empty array (return 0), all duplicates, no duplicates.
    """
    if not arr:
        return 0
    i = 0
    for j in range(1, len(arr)):
        if arr[j] != arr[i]:
            i += 1
            arr[i] = arr[j]
    return i + 1

def find_pairs(arr, target):
    """
    Find pairs that sum to target using two-pointer technique on a sorted array.

    Concept: Sort the array, then use two pointers from start and end.
    Time Complexity: O(n log n) due to sorting, O(n) for the pointers.
    Space Complexity: O(1) extra space if sorting in-place.
    ML/DL Implications: Finding complementary features or pairs in collaborative filtering.
    Example Use Case: Identifying pairs of features that sum to a threshold.
    Edge Cases: Empty array, no pairs found, multiple pairs.
    """
    arr.sort()
    left, right = 0, len(arr) - 1
    pairs = []
    while left < right:
        current_sum = arr[left] + arr[right]
        if current_sum == target:
            pairs.append((arr[left], arr[right]))
            left += 1
            right -= 1
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    return pairs

def visualize_array(arr, title):
    """
    Simple visualization of array states using Matplotlib.
    """
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(arr)), arr, color='blue')
    plt.title(title)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.show()

def visualize_complexity():
    """
    Visualize time complexity comparison: O(n) vs O(log n).
    """
    n = np.arange(1, 101)
    plt.figure(figsize=(8, 4))
    plt.plot(n, n, label='O(n)', color='red')
    plt.plot(n, np.log2(n), label='O(log n)', color='green')
    plt.title('Time Complexity Comparison')
    plt.xlabel('n')
    plt.ylabel('Operations')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Test DynamicArray
    da = DynamicArray()
    for i in range(10):
        da.append(i)
    print("Dynamic Array:", da.get_array())

    # Test Binary Search
    sorted_arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    print("Binary Search for 5:", binary_search(sorted_arr, 5))
    print("Binary Search for 11:", binary_search(sorted_arr, 11))

    # Test Remove Duplicates
    dup_arr = [1, 1, 2, 2, 3, 4, 4, 5]
    new_length = remove_duplicates(dup_arr)
    print("Array after removing duplicates:", dup_arr[:new_length])

    # Test Find Pairs
    pair_arr = [1, 2, 3, 4, 5, 6]
    pairs = find_pairs(pair_arr, 7)
    print("Pairs summing to 7:", pairs)

    # Visualizations
    visualize_array(sorted_arr, "Sorted Array")
    visualize_complexity()