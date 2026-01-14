import matplotlib.pyplot as plt

class Node:
    """
    Node class for linked list in chaining hash table.
    """
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.next = None

class HashTableChaining:
    """
    Hash table with chaining using linked lists.

    Concepts: Each bucket contains a linked list to handle collisions.
    Time Complexity: Average O(1) for insert, search, delete; Worst O(n) due to collisions.
    Space Complexity: O(n) for storing elements.
    ML/DL Implications: Used for fast lookups in feature maps, e.g., in NLP for word-to-index mappings or in recommendation systems for user-item interactions.
    Examples: Storing word frequencies in text analysis.
    Edge Cases: High load factor leads to long chains, triggering resize; collisions are common with poor hash functions.
    """
    def __init__(self, size=10):
        self.size = size
        self.table = [None] * size
        self.count = 0

    def _hash(self, key):
        # Custom hash function: simple modulo hash
        return hash(key) % self.size

    def insert(self, key, value):
        index = self._hash(key)
        if self.table[index] is None:
            self.table[index] = Node(key, value)
        else:
            current = self.table[index]
            while current:
                if current.key == key:
                    current.value = value
                    return
                if current.next is None:
                    break
                current = current.next
            current.next = Node(key, value)
        self.count += 1
        if self.load_factor() > 0.75:
            self._resize()

    def search(self, key):
        index = self._hash(key)
        current = self.table[index]
        while current:
            if current.key == key:
                return current.value
            current = current.next
        return None

    def delete(self, key):
        index = self._hash(key)
        current = self.table[index]
        prev = None
        while current:
            if current.key == key:
                if prev:
                    prev.next = current.next
                else:
                    self.table[index] = current.next
                self.count -= 1
                return True
            prev = current
            current = current.next
        return False

    def load_factor(self):
        return self.count / self.size

    def _resize(self):
        old_table = self.table
        self.size *= 2
        self.table = [None] * self.size
        self.count = 0
        for head in old_table:
            current = head
            while current:
                self.insert(current.key, current.value)
                current = current.next

    def get_chain_lengths(self):
        lengths = []
        for head in self.table:
            length = 0
            current = head
            while current:
                length += 1
                current = current.next
            lengths.append(length)
        return lengths

class HashTableOpenAddressing:
    """
    Hash table with open addressing using linear probing.

    Concepts: On collision, probe next slots linearly until empty slot found.
    Time Complexity: Average O(1); Worst O(n) with clustering.
    Space Complexity: O(n).
    ML/DL Implications: Efficient for dense data structures in ML, like hash-based indexing in databases used for feature storage.
    Examples: Implementing sets or maps for quick access in data preprocessing.
    Edge Cases: Clustering can degrade performance; resizing helps; deletion uses tombstones to avoid breaking probes.
    """
    def __init__(self, size=10):
        self.size = size
        self.table = [None] * size
        self.count = 0

    def _hash(self, key):
        # Custom hash function: djb2 variant
        h = 5381
        for char in str(key):
            h = ((h << 5) + h) + ord(char)
        return h % self.size

    def insert(self, key, value):
        index = self._hash(key)
        original_index = index
        while self.table[index] is not None and self.table[index] != ('DELETED', None):
            if self.table[index][0] == key:
                self.table[index] = (key, value)
                return
            index = (index + 1) % self.size
            if index == original_index:
                raise Exception("Hash table full")
        self.table[index] = (key, value)
        self.count += 1
        if self.load_factor() > 0.75:
            self._resize()

    def search(self, key):
        index = self._hash(key)
        original_index = index
        while self.table[index] is not None:
            if self.table[index] != ('DELETED', None) and self.table[index][0] == key:
                return self.table[index][1]
            index = (index + 1) % self.size
            if index == original_index:
                break
        return None

    def delete(self, key):
        index = self._hash(key)
        original_index = index
        while self.table[index] is not None:
            if self.table[index] != ('DELETED', None) and self.table[index][0] == key:
                self.table[index] = ('DELETED', None)
                self.count -= 1
                return True
            index = (index + 1) % self.size
            if index == original_index:
                break
        return False

    def load_factor(self):
        return self.count / self.size

    def _resize(self):
        old_table = self.table
        self.size *= 2
        self.table = [None] * self.size
        self.count = 0
        for item in old_table:
            if item and item[0] != 'DELETED':
                self.insert(item[0], item[1])

if __name__ == "__main__":
    # Example for HashTableChaining
    ht_chain = HashTableChaining(10)
    words = ["apple", "banana", "cherry", "apple", "date", "elderberry", "fig", "grape", "honeydew", "apple"]
    load_factors = []
    for word in words:
        ht_chain.insert(word, ht_chain.search(word) + 1 if ht_chain.search(word) else 1)
        load_factors.append(ht_chain.load_factor())

    print("Word frequencies:")
    for word in set(words):
        print(f"{word}: {ht_chain.search(word)}")

    # Visualization: Load factor over inserts
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(len(load_factors)), load_factors, marker='o')
    plt.title('Load Factor Over Inserts (Chaining)')
    plt.xlabel('Insert Number')
    plt.ylabel('Load Factor')

    # Chain lengths distribution
    chain_lengths = ht_chain.get_chain_lengths()
    plt.subplot(1, 2, 2)
    plt.bar(range(len(chain_lengths)), chain_lengths)
    plt.title('Chain Lengths Distribution (Chaining)')
    plt.xlabel('Bucket Index')
    plt.ylabel('Chain Length')
    plt.tight_layout()
    plt.show()

    # Example for HashTableOpenAddressing
    ht_open = HashTableOpenAddressing(10)
    load_factors_open = []
    for word in words:
        ht_open.insert(word, ht_open.search(word) + 1 if ht_open.search(word) else 1)
        load_factors_open.append(ht_open.load_factor())

    print("\nWord frequencies (Open Addressing):")
    for word in set(words):
        print(f"{word}: {ht_open.search(word)}")

    # Visualization: Load factor over inserts
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(len(load_factors_open)), load_factors_open, marker='o')
    plt.title('Load Factor Over Inserts (Open Addressing)')
    plt.xlabel('Insert Number')
    plt.ylabel('Load Factor')

    # Occupied slots (simple histogram of non-None)
    occupied = [1 if slot is not None and slot != ('DELETED', None) else 0 for slot in ht_open.table]
    plt.subplot(1, 2, 2)
    plt.bar(range(len(occupied)), occupied)
    plt.title('Occupied Slots (Open Addressing)')
    plt.xlabel('Slot Index')
    plt.ylabel('Occupied (1/0)')
    plt.tight_layout()
    plt.show()