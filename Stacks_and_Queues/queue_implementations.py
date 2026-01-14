from collections import deque
import heapq
import matplotlib.pyplot as plt

class Node:
    """
    Node class for linked list implementation.
    Each node holds data and a reference to the next node.
    """
    def __init__(self, data):
        self.data = data
        self.next = None

class QueueList:
    """
    Queue implementation using Python list.
    A queue is a FIFO (First In First Out) data structure.
    Operations:
    - Enqueue: O(1)
    - Dequeue: O(n) due to list.pop(0)
    - Front: O(1)
    - Is_empty: O(1)
    - Size: O(1)
    Space: O(n)
    Note: Inefficient for large queues; use deque instead.
    ML/DL implications: Queues are used for task scheduling in ML pipelines, BFS in graph algorithms for model training data flow.
    Examples: Print queue, breadth-first search.
    Edge cases: Dequeue/front on empty queue raises IndexError.
    """
    def __init__(self):
        self.queue = []

    def enqueue(self, item):
        self.queue.append(item)

    def dequeue(self):
        if not self.is_empty():
            return self.queue.pop(0)
        raise IndexError("Dequeue from empty queue")

    def front(self):
        if not self.is_empty():
            return self.queue[0]
        raise IndexError("Front from empty queue")

    def is_empty(self):
        return len(self.queue) == 0

    def size(self):
        return len(self.queue)

class QueueLinkedList:
    """
    Queue implementation using linked list.
    Advantages: Efficient enqueue/dequeue O(1).
    Operations:
    - Enqueue: O(1)
    - Dequeue: O(1)
    - Front: O(1)
    - Is_empty: O(1)
    - Size: O(1)
    Space: O(n)
    ML/DL implications: Useful for streaming data processing in real-time ML, like online learning queues.
    Examples: CPU scheduling, message queues in distributed systems.
    Edge cases: Dequeue/front on empty queue raises IndexError.
    """
    def __init__(self):
        self.front = None
        self.rear = None
        self._size = 0

    def enqueue(self, item):
        new_node = Node(item)
        if self.rear:
            self.rear.next = new_node
        self.rear = new_node
        if not self.front:
            self.front = new_node
        self._size += 1

    def dequeue(self):
        if self.is_empty():
            raise IndexError("Dequeue from empty queue")
        item = self.front.data
        self.front = self.front.next
        if not self.front:
            self.rear = None
        self._size -= 1
        return item

    def front(self):
        if self.is_empty():
            raise IndexError("Front from empty queue")
        return self.front.data

    def is_empty(self):
        return self.front is None

    def size(self):
        return self._size

class Deque:
    """
    Double-ended queue using collections.deque.
    Allows operations from both ends.
    Operations:
    - Add_front/Remove_front: O(1)
    - Add_rear/Remove_rear: O(1)
    - Front/Rear: O(1)
    - Is_empty/Size: O(1)
    Space: O(n)
    ML/DL implications: Used in sliding window algorithms for signal processing, deque for efficient feature extraction in time series.
    Examples: Palindrome checking, sliding window maximum.
    Edge cases: Operations on empty deque raise IndexError.
    """
    def __init__(self):
        self.deque = deque()

    def add_front(self, item):
        self.deque.appendleft(item)

    def add_rear(self, item):
        self.deque.append(item)

    def remove_front(self):
        if not self.is_empty():
            return self.deque.popleft()
        raise IndexError("Remove from empty deque")

    def remove_rear(self):
        if not self.is_empty():
            return self.deque.pop()
        raise IndexError("Remove from empty deque")

    def front(self):
        if not self.is_empty():
            return self.deque[0]
        raise IndexError("Front from empty deque")

    def rear(self):
        if not self.is_empty():
            return self.deque[-1]
        raise IndexError("Rear from empty deque")

    def is_empty(self):
        return len(self.deque) == 0

    def size(self):
        return len(self.deque)

class PriorityQueue:
    """
    Priority queue using heapq (min-heap).
    Items are dequeued based on priority (lowest first).
    Operations:
    - Push: O(log n)
    - Pop: O(log n)
    - Peek: O(1)
    - Is_empty/Size: O(1)
    Space: O(n)
    ML/DL implications: Used for priority-based sampling in reinforcement learning, task prioritization in ML workflows.
    Examples: Dijkstra's algorithm, Huffman coding.
    Edge cases: Pop/peek on empty queue raises IndexError.
    """
    def __init__(self):
        self.heap = []

    def push(self, item, priority):
        heapq.heappush(self.heap, (priority, item))

    def pop(self):
        if not self.is_empty():
            return heapq.heappop(self.heap)[1]
        raise IndexError("Pop from empty priority queue")

    def peek(self):
        if not self.is_empty():
            return self.heap[0][1]
        raise IndexError("Peek from empty priority queue")

    def is_empty(self):
        return len(self.heap) == 0

    def size(self):
        return len(self.heap)

def sliding_window_maximum(nums, k):
    """
    Finds the maximum in each sliding window of size k using deque.
    Deque maintains indices of potential maxima in decreasing order.
    Time: O(n)
    Space: O(k)
    ML/DL implications: Essential for feature engineering in time series data, like moving averages in signal processing for anomaly detection.
    Examples: nums=[1,3,-1,-3,5,3,6,7], k=3 -> [3,3,5,5,6,7]
    Edge cases: k > len(nums) (empty result), k=1 (each element), empty nums.
    """
    if not nums or k == 0:
        return []
    dq = deque()
    result = []
    for i in range(len(nums)):
        while dq and nums[dq[-1]] <= nums[i]:
            dq.pop()
        dq.append(i)
        if dq[0] == i - k:
            dq.popleft()
        if i >= k - 1:
            result.append(nums[dq[0]])
    return result

def visualize_sliding_window(nums, k):
    """
    Visualizes sliding window maximum using matplotlib.
    Plots the array, highlights windows, and marks maxima.
    """
    maxes = sliding_window_maximum(nums, k)
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(nums)), nums, marker='o', linestyle='-', color='blue', label='Array Elements')
    for i in range(len(maxes)):
        start = i
        end = min(i + k - 1, len(nums) - 1)
        plt.axvspan(start, end, alpha=0.3, color='yellow', label='Window' if i == 0 else "")
        plt.scatter(end, maxes[i], color='red', s=100, edgecolor='black', label='Max in Window' if i == 0 else "")
    plt.title(f'Sliding Window Maximum (k={k})')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    print("Demonstrating Queue Implementations")
    print("=" * 40)

    print("\nQueue using list:")
    q = QueueList()
    q.enqueue(1)
    q.enqueue(2)
    q.enqueue(3)
    print(f"Dequeue: {q.dequeue()}")
    print(f"Front: {q.front()}")
    print(f"Size: {q.size()}")

    print("\nQueue using linked list:")
    ql = QueueLinkedList()
    ql.enqueue(1)
    ql.enqueue(2)
    ql.enqueue(3)
    print(f"Dequeue: {ql.dequeue()}")
    print(f"Front: {ql.front()}")
    print(f"Size: {ql.size()}")

    print("\nDeque:")
    d = Deque()
    d.add_rear(1)
    d.add_front(2)
    d.add_rear(3)
    print(f"Remove front: {d.remove_front()}")
    print(f"Remove rear: {d.remove_rear()}")
    print(f"Front: {d.front()}")
    print(f"Rear: {d.rear()}")

    print("\nPriority Queue:")
    pq = PriorityQueue()
    pq.push('task1', 3)
    pq.push('task2', 1)
    pq.push('task3', 2)
    print(f"Pop: {pq.pop()}")
    print(f"Peek: {pq.peek()}")

    print("\nSliding window maximum:")
    nums = [1, 3, -1, -3, 5, 3, 6, 7]
    k = 3
    print(f"Array: {nums}")
    print(f"Max in windows of size {k}: {sliding_window_maximum(nums, k)}")

    print("\nVisualizing sliding window maximum...")
    visualize_sliding_window(nums, k)