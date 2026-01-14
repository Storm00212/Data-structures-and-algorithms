"""
Linked List Basics Implementation

This module implements basic linked list structures and operations relevant to Machine Learning and Deep Learning.
Linked lists are useful for dynamic data structures where insertions and deletions are frequent, such as:
- Sparse matrix representations in ML (e.g., CSR format uses linked list concepts)
- Efficient handling of dynamic feature lists in streaming ML
- Memory-efficient storage for variable-length sequences

Time complexities:
- Insert at head: O(1)
- Insert at tail: O(n) for singly, O(1) with tail pointer
- Delete: O(1) if node known, O(n) to find
- Search: O(n)

Space: O(n) for n nodes
"""

import networkx as nx
import matplotlib.pyplot as plt

class Node:
    """Basic node for singly linked list."""
    def __init__(self, data):
        self.data = data
        self.next = None

class DoublyNode:
    """Node for doubly linked list with prev pointer."""
    def __init__(self, data):
        self.data = data
        self.next = None
        self.prev = None

class SinglyLinkedList:
    """Singly Linked List implementation.

    ML/DL implications: Used for sparse data where only non-zero elements are stored,
    reducing memory in large datasets. Example: Feature vectors in NLP preprocessing.
    """
    def __init__(self):
        self.head = None

    def insert_at_head(self, data):
        """Insert at head: O(1) time."""
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node

    def insert_at_tail(self, data):
        """Insert at tail: O(n) time without tail pointer."""
        if not self.head:
            self.head = Node(data)
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = Node(data)

    def delete(self, data):
        """Delete first occurrence: O(n) time."""
        if not self.head:
            return
        if self.head.data == data:
            self.head = self.head.next
            return
        current = self.head
        while current.next and current.next.data != data:
            current = current.next
        if current.next:
            current.next = current.next.next

    def search(self, data):
        """Search for data: O(n) time."""
        current = self.head
        while current:
            if current.data == data:
                return True
            current = current.next
        return False

    def display(self):
        """Display list."""
        current = self.head
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("None")

    def visualize(self):
        """Visualize using NetworkX."""
        G = nx.DiGraph()
        current = self.head
        pos = 0
        while current:
            G.add_node(pos, label=str(current.data))
            if current.next:
                G.add_edge(pos, pos + 1)
            current = current.next
            pos += 1
        labels = nx.get_node_attributes(G, 'label')
        nx.draw(G, labels=labels, with_labels=True, node_color='lightblue', arrows=True)
        plt.show()

class DoublyLinkedList:
    """Doubly Linked List with bidirectional traversal.

    ML/DL implications: Useful for bidirectional RNNs or LSTMs where forward and backward passes are needed.
    Example: Managing sequences in time-series data with easy reversal.
    """
    def __init__(self):
        self.head = None
        self.tail = None

    def insert_at_head(self, data):
        """Insert at head: O(1)."""
        new_node = DoublyNode(data)
        if not self.head:
            self.head = self.tail = new_node
        else:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node

    def insert_at_tail(self, data):
        """Insert at tail: O(1) with tail pointer."""
        new_node = DoublyNode(data)
        if not self.tail:
            self.head = self.tail = new_node
        else:
            self.tail.next = new_node
            new_node.prev = self.tail
            self.tail = new_node

    def delete(self, data):
        """Delete first occurrence: O(n)."""
        current = self.head
        while current:
            if current.data == data:
                if current.prev:
                    current.prev.next = current.next
                else:
                    self.head = current.next
                if current.next:
                    current.next.prev = current.prev
                else:
                    self.tail = current.prev
                return
            current = current.next

    def search(self, data):
        """Search: O(n)."""
        current = self.head
        while current:
            if current.data == data:
                return True
            current = current.next
        return False

    def display_forward(self):
        """Display forward."""
        current = self.head
        while current:
            print(current.data, end=" <-> ")
            current = current.next
        print("None")

    def display_backward(self):
        """Display backward."""
        current = self.tail
        while current:
            print(current.data, end=" <-> ")
            current = current.prev
        print("None")

    def visualize(self):
        """Visualize bidirectional."""
        G = nx.DiGraph()
        current = self.head
        pos = 0
        while current:
            G.add_node(pos, label=str(current.data))
            if current.next:
                G.add_edge(pos, pos + 1)
                G.add_edge(pos + 1, pos)  # Bidirectional
            current = current.next
            pos += 1
        labels = nx.get_node_attributes(G, 'label')
        nx.draw(G, labels=labels, with_labels=True, node_color='lightgreen', arrows=True)
        plt.show()

class CircularLinkedList:
    """Circular Linked List for cyclic data.

    ML/DL implications: Useful for cyclic buffers in streaming data, or circular queues in online learning.
    Example: Rotating feature sets in continual learning.
    """
    def __init__(self):
        self.head = None

    def insert_at_head(self, data):
        """Insert at head: O(1)."""
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            self.head.next = self.head
        else:
            new_node.next = self.head.next
            self.head.next = new_node
            # Swap data to make new head
            self.head.data, new_node.data = new_node.data, self.head.data

    def insert_at_tail(self, data):
        """Insert at tail: O(n)."""
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            self.head.next = self.head
        else:
            current = self.head
            while current.next != self.head:
                current = current.next
            current.next = new_node
            new_node.next = self.head

    def delete(self, data):
        """Delete: O(n)."""
        if not self.head:
            return
        current = self.head
        prev = None
        while True:
            if current.data == data:
                if prev:
                    prev.next = current.next
                else:
                    # Deleting head
                    if current.next == self.head:
                        self.head = None
                    else:
                        self.head = current.next
                return
            prev = current
            current = current.next
            if current == self.head:
                break

    def search(self, data):
        """Search: O(n)."""
        if not self.head:
            return False
        current = self.head
        while True:
            if current.data == data:
                return True
            current = current.next
            if current == self.head:
                break
        return False

    def display(self):
        """Display circular list."""
        if not self.head:
            print("Empty")
            return
        current = self.head
        while True:
            print(current.data, end=" -> ")
            current = current.next
            if current == self.head:
                break
        print("(back to head)")

    def visualize(self):
        """Visualize circular."""
        G = nx.DiGraph()
        if not self.head:
            return
        current = self.head
        pos = 0
        start_pos = pos
        while True:
            G.add_node(pos, label=str(current.data))
            next_pos = pos + 1 if current.next != self.head else start_pos
            G.add_edge(pos, next_pos)
            current = current.next
            pos += 1
            if current == self.head:
                break
        labels = nx.get_node_attributes(G, 'label')
        nx.draw(G, labels=labels, with_labels=True, node_color='lightcoral', arrows=True)
        plt.show()

if __name__ == "__main__":
    # Test SinglyLinkedList
    sll = SinglyLinkedList()
    sll.insert_at_head(1)
    sll.insert_at_head(2)
    sll.insert_at_tail(3)
    sll.display()  # 2 -> 1 -> 3 -> None
    print("Search 1:", sll.search(1))
    sll.delete(1)
    sll.display()  # 2 -> 3 -> None
    # sll.visualize()  # Uncomment to visualize

    # Test DoublyLinkedList
    dll = DoublyLinkedList()
    dll.insert_at_head(1)
    dll.insert_at_tail(2)
    dll.insert_at_head(0)
    dll.display_forward()  # 0 <-> 1 <-> 2 <-> None
    dll.display_backward()  # 2 <-> 1 <-> 0 <-> None
    dll.delete(1)
    dll.display_forward()  # 0 <-> 2 <-> None
    # dll.visualize()  # Uncomment

    # Test CircularLinkedList
    cll = CircularLinkedList()
    cll.insert_at_head(1)
    cll.insert_at_tail(2)
    cll.insert_at_tail(3)
    cll.display()  # 1 -> 2 -> 3 -> (back to head)
    cll.delete(2)
    cll.display()  # 1 -> 3 -> (back to head)
    # cll.visualize()  # Uncomment