"""
Linked List Algorithms Implementation

This module implements advanced algorithms using linked lists, with applications in ML/DL.
"""

import networkx as nx
import matplotlib.pyplot as plt

class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class DoublyNode:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.prev = None

def detect_cycle(head):
    """
    Cycle detection using Floyd's Tortoise and Hare algorithm.

    Time: O(n), Space: O(1)
    ML/DL applications: Detecting cycles in graph-based preprocessing for ML pipelines,
    e.g., ensuring no loops in dependency graphs for model training.
    Edge cases: Empty list (no cycle), single node (no cycle), cycle at start.
    """
    if not head or not head.next:
        return False
    slow = head
    fast = head.next
    while fast and fast.next:
        if slow == fast:
            return True
        slow = slow.next
        fast = fast.next.next
    return False

def merge_sorted_lists(l1, l2):
    """
    Merge two sorted linked lists.

    Time: O(n + m), Space: O(1) extra space.
    ML/DL applications: Merging sorted feature lists in ensemble methods or data preprocessing.
    Edge cases: One list empty, both empty.
    """
    dummy = Node(0)
    current = dummy
    while l1 and l2:
        if l1.data < l2.data:
            current.next = l1
            l1 = l1.next
        else:
            current.next = l2
            l2 = l2.next
        current = current.next
    current.next = l1 or l2
    return dummy.next

def visualize_list(head, title="Linked List"):
    """Visualize a singly linked list."""
    G = nx.DiGraph()
    current = head
    pos = 0
    while current:
        G.add_node(pos, label=str(current.data))
        if current.next:
            G.add_edge(pos, pos + 1)
        current = current.next
        pos += 1
    labels = nx.get_node_attributes(G, 'label')
    nx.draw(G, labels=labels, with_labels=True, node_color='lightblue', arrows=True)
    plt.title(title)
    plt.show()

class LRUCache:
    """
    LRU Cache using doubly linked list and hash map.

    Time: O(1) for get and put.
    Space: O(capacity)
    ML/DL applications: Caching model weights or intermediate computations in memory-constrained environments,
    e.g., in federated learning or edge ML.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.head = DoublyNode(0)  # Dummy head
        self.tail = DoublyNode(0)  # Dummy tail
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_to_front(self, node):
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def get(self, key):
        if key in self.cache:
            node = self.cache[key]
            self._remove(node)
            self._add_to_front(node)
            return node.data
        return -1

    def put(self, key, value):
        if key in self.cache:
            self._remove(self.cache[key])
        node = DoublyNode(value)
        self.cache[key] = node
        self._add_to_front(node)
        if len(self.cache) > self.capacity:
            lru = self.tail.prev
            self._remove(lru)
            del self.cache[list(self.cache.keys())[list(self.cache.values()).index(lru)]]

    def visualize(self):
        """Visualize current cache order."""
        G = nx.DiGraph()
        current = self.head.next
        pos = 0
        while current != self.tail:
            G.add_node(pos, label=str(current.data))
            if current.next != self.tail:
                G.add_edge(pos, pos + 1)
            current = current.next
            pos += 1
        labels = nx.get_node_attributes(G, 'label')
        nx.draw(G, labels=labels, with_labels=True, node_color='lightgreen', arrows=True)
        plt.title("LRU Cache Order")
        plt.show()

if __name__ == "__main__":
    # Test cycle detection
    head = Node(1)
    head.next = Node(2)
    head.next.next = Node(3)
    head.next.next.next = head.next  # Cycle
    print("Cycle detected:", detect_cycle(head))

    # Test merge
    l1 = Node(1)
    l1.next = Node(3)
    l2 = Node(2)
    l2.next = Node(4)
    merged = merge_sorted_lists(l1, l2)
    visualize_list(merged, "Merged Sorted List")

    # Test LRU Cache
    cache = LRUCache(2)
    cache.put(1, 1)
    cache.put(2, 2)
    print("Get 1:", cache.get(1))  # Should move 1 to front
    cache.put(3, 3)  # Evict 2
    print("Get 2:", cache.get(2))  # -1
    # cache.visualize()  # Uncomment