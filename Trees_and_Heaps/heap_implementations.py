import matplotlib.pyplot as plt
import heapq  # For comparison in PriorityQueue, but implementing custom heaps

class Heap:
    """
    Base Heap class. Heaps are complete binary trees stored in arrays.
    Time complexity: Insert O(log n), Extract O(log n), Build heap O(n).
    Space complexity: O(n).
    ML/DL applications: Heaps used in k-nearest neighbors for maintaining top-k elements,
    priority queues in algorithms like Dijkstra for shortest paths in graph-based ML.
    Edge cases: Empty heap, single element, handling duplicates.
    """
    def __init__(self):
        self.heap = []

    def parent(self, i):
        return (i - 1) // 2

    def left_child(self, i):
        return 2 * i + 1

    def right_child(self, i):
        return 2 * i + 2

    def swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]

    def heapify_up(self, i):
        while i > 0 and self.compare(self.heap[self.parent(i)], self.heap[i]):
            self.swap(i, self.parent(i))
            i = self.parent(i)

    def heapify_down(self, i):
        size = len(self.heap)
        while True:
            left = self.left_child(i)
            right = self.right_child(i)
            smallest_or_largest = i
            if left < size and self.compare(self.heap[smallest_or_largest], self.heap[left]):
                smallest_or_largest = left
            if right < size and self.compare(self.heap[smallest_or_largest], self.heap[right]):
                smallest_or_largest = right
            if smallest_or_largest == i:
                break
            self.swap(i, smallest_or_largest)
            i = smallest_or_largest

    def insert(self, value):
        self.heap.append(value)
        self.heapify_up(len(self.heap) - 1)

    def extract(self):
        if not self.heap:
            return None
        if len(self.heap) == 1:
            return self.heap.pop()
        root = self.heap[0]
        self.heap[0] = self.heap.pop()
        self.heapify_down(0)
        return root

    def peek(self):
        return self.heap[0] if self.heap else None

    def compare(self, a, b):
        # To be overridden
        pass

class MinHeap(Heap):
    """
    MinHeap: Parent <= children.
    Used for finding minimum elements efficiently.
    """
    def compare(self, a, b):
        return a > b  # For min heap, swap if parent > child

class MaxHeap(Heap):
    """
    MaxHeap: Parent >= children.
    Used for finding maximum elements efficiently.
    """
    def compare(self, a, b):
        return a < b  # For max heap, swap if parent < child

def heap_sort(arr, ascending=True):
    """
    Heap sort algorithm.
    Time complexity: O(n log n).
    Space complexity: O(1) extra space (in-place).
    ML applications: Sorting features or predictions in ML pipelines.
    Example: Sorting a list of prediction scores.
    Edge cases: Empty array, single element, already sorted.
    """
    if ascending:
        heap = MinHeap()
    else:
        heap = MaxHeap()
    for num in arr:
        heap.insert(num)
    sorted_arr = []
    while heap.heap:
        sorted_arr.append(heap.extract())
    return sorted_arr

class PriorityQueue:
    """
    Priority Queue implemented using MinHeap.
    Supports insert and extract_min.
    Time complexity: Insert O(log n), Extract O(log n).
    ML applications: Managing events in simulations, or priority-based sampling in ML.
    """
    def __init__(self):
        self.heap = MinHeap()

    def insert(self, value):
        self.heap.insert(value)

    def extract_min(self):
        return self.heap.extract()

    def peek_min(self):
        return self.heap.peek()

    def is_empty(self):
        return len(self.heap.heap) == 0

def visualize_heap(heap, title="Heap Array"):
    """
    Visualize the heap as a bar chart using Matplotlib.
    """
    if not heap.heap:
        print("Heap is empty.")
        return
    plt.bar(range(len(heap.heap)), heap.heap, color='skyblue')
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.show()

def visualize_heap_sort_process(arr):
    """
    Visualize heap sort by showing initial and sorted array.
    """
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.bar(range(len(arr)), arr, color='lightcoral')
    plt.title("Original Array")
    plt.subplot(1, 2, 2)
    sorted_arr = heap_sort(arr.copy())
    plt.bar(range(len(sorted_arr)), sorted_arr, color='lightgreen')
    plt.title("Sorted Array (Heap Sort)")
    plt.show()

if __name__ == "__main__":
    # Example: MinHeap and MaxHeap
    min_heap = MinHeap()
    data = [10, 20, 15, 30, 40]
    for d in data:
        min_heap.insert(d)
    print("MinHeap extract min:", min_heap.extract())
    visualize_heap(min_heap, "MinHeap after extract")

    max_heap = MaxHeap()
    for d in data:
        max_heap.insert(d)
    print("MaxHeap extract max:", max_heap.extract())
    visualize_heap(max_heap, "MaxHeap after extract")

    # Example: Heap sort
    arr = [64, 34, 25, 12, 22, 11, 90]
    print("Original array:", arr)
    sorted_arr = heap_sort(arr)
    print("Sorted array:", sorted_arr)
    visualize_heap_sort_process(arr)

    # Example: Priority Queue
    pq = PriorityQueue()
    priorities = [5, 1, 3, 2, 4]
    for p in priorities:
        pq.insert(p)
    print("Priority Queue extract min:", pq.extract_min())
    print("Next min:", pq.extract_min())