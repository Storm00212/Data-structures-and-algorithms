import matplotlib.pyplot as plt

class BloomFilter:
    """
    Bloom filter for approximate set membership.

    Concepts: Uses multiple hash functions to set bits in a bit array; checks if all bits are set for membership.
    Time Complexity: O(k) for add/check, where k is number of hash functions.
    Space Complexity: O(m), where m is bit array size.
    ML/DL Implications: Used for efficient approximate membership testing in large datasets, e.g., checking if a feature is in a set without storing all, useful in streaming data or caching.
    Examples: Checking if a URL is in a blacklist; deduplication in data pipelines.
    Edge Cases: False positives possible; no false negatives; optimal size and k depend on expected elements and false positive rate.
    """
    def __init__(self, size, hash_count):
        self.size = size
        self.hash_count = hash_count
        self.bit_array = [0] * size

    def _hashes(self, item):
        hashes = []
        for i in range(self.hash_count):
            h = hash(str(item) + str(i)) % self.size
            hashes.append(h)
        return hashes

    def add(self, item):
        for h in self._hashes(item):
            self.bit_array[h] = 1

    def check(self, item):
        for h in self._hashes(item):
            if self.bit_array[h] == 0:
                return False
        return True

class CuckooHashing:
    """
    Cuckoo hashing for fast lookups with two tables.

    Concepts: Each key hashes to two possible positions; on collision, evicts the existing key to its alternative position.
    Time Complexity: Average O(1) for operations; worst case can be higher due to eviction loops.
    Space Complexity: O(n), with higher constant factor due to two tables.
    ML/DL Implications: Provides worst-case O(1) lookups, useful for real-time systems in ML, like fast feature indexing.
    Examples: Implementing associative arrays with guaranteed lookup time.
    Edge Cases: Eviction cycles can occur; resizing or rehashing needed; not suitable for very high load factors.
    """
    def __init__(self, size):
        self.size = size
        self.table1 = [None] * size
        self.table2 = [None] * size

    def _hash1(self, key):
        return hash(key) % self.size

    def _hash2(self, key):
        return (hash(key) * 31 + 7) % self.size

    def insert(self, key, value):
        if self.search(key) is not None:
            # Update
            if self.table1[self._hash1(key)] and self.table1[self._hash1(key)][0] == key:
                self.table1[self._hash1(key)] = (key, value)
            else:
                self.table2[self._hash2(key)] = (key, value)
            return
        # Insert
        key_to_insert = key
        value_to_insert = value
        for _ in range(self.size):  # Limit evictions
            index1 = self._hash1(key_to_insert)
            if self.table1[index1] is None:
                self.table1[index1] = (key_to_insert, value_to_insert)
                return
            # Evict from table1
            evicted_key, evicted_value = self.table1[index1]
            self.table1[index1] = (key_to_insert, value_to_insert)
            key_to_insert, value_to_insert = evicted_key, evicted_value
            # Now to table2
            index2 = self._hash2(key_to_insert)
            if self.table2[index2] is None:
                self.table2[index2] = (key_to_insert, value_to_insert)
                return
            # Evict from table2
            evicted_key, evicted_value = self.table2[index2]
            self.table2[index2] = (key_to_insert, value_to_insert)
            key_to_insert, value_to_insert = evicted_key, evicted_value
        raise Exception("Cuckoo insert failed: possible cycle")

    def search(self, key):
        index1 = self._hash1(key)
        if self.table1[index1] and self.table1[index1][0] == key:
            return self.table1[index1][1]
        index2 = self._hash2(key)
        if self.table2[index2] and self.table2[index2][0] == key:
            return self.table2[index2][1]
        return None

    def delete(self, key):
        index1 = self._hash1(key)
        if self.table1[index1] and self.table1[index1][0] == key:
            self.table1[index1] = None
            return True
        index2 = self._hash2(key)
        if self.table2[index2] and self.table2[index2][0] == key:
            self.table2[index2] = None
            return True
        return False

class FeatureHashing:
    """
    Feature hashing for dimensionality reduction in ML.

    Concepts: Hashes features to a fixed-size vector, accumulating counts or values.
    Time Complexity: O(d) for d features, but effective O(1) per feature.
    Space Complexity: O(m), where m is hash size, independent of feature count.
    ML/DL Implications: Reduces high-dimensional sparse features to fixed size, useful in text classification or recommendation systems to handle large vocabularies.
    Examples: Hashing word counts in bag-of-words for text data.
    Edge Cases: Collisions can merge features; choose large m to minimize; signed hashing can handle negative weights.
    """
    def __init__(self, size):
        self.size = size

    def hash_feature(self, feature):
        return hash(feature) % self.size

    def vectorize(self, features):
        vec = [0] * self.size
        for f, val in features.items():
            idx = self.hash_feature(f)
            vec[idx] += val
        return vec

if __name__ == "__main__":
    # Bloom Filter Example
    bf = BloomFilter(100, 3)
    items = ["apple", "banana", "cherry"]
    for item in items:
        bf.add(item)

    print("Bloom Filter Check:")
    print(f"apple in filter: {bf.check('apple')}")
    print(f"date in filter: {bf.check('date')}")  # False positive possible

    # Visualization: Bit array set bits
    set_bits = sum(bf.bit_array)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.bar(range(len(bf.bit_array)), bf.bit_array)
    plt.title(f'Bloom Filter Bit Array (Set bits: {set_bits})')
    plt.xlabel('Bit Index')
    plt.ylabel('Set (1/0)')

    # Cuckoo Hashing Example
    ch = CuckooHashing(10)
    pairs = [("key1", 1), ("key2", 2), ("key3", 3)]
    for k, v in pairs:
        ch.insert(k, v)

    print("\nCuckoo Hashing Search:")
    for k, _ in pairs:
        print(f"{k}: {ch.search(k)}")

    # Visualization: Occupied slots in tables
    occupied1 = [1 if slot else 0 for slot in ch.table1]
    occupied2 = [1 if slot else 0 for slot in ch.table2]
    plt.subplot(1, 3, 2)
    plt.bar(range(len(occupied1)), occupied1, label='Table 1')
    plt.bar(range(len(occupied2)), occupied2, bottom=occupied1, label='Table 2')
    plt.title('Cuckoo Tables Occupancy')
    plt.xlabel('Index')
    plt.ylabel('Occupied')
    plt.legend()

    # Feature Hashing Example
    fh = FeatureHashing(10)
    features = {"word1": 2, "word2": 1, "word3": 3}
    vec = fh.vectorize(features)
    print(f"\nFeature Vector: {vec}")

    # Visualization: Feature vector
    plt.subplot(1, 3, 3)
    plt.bar(range(len(vec)), vec)
    plt.title('Feature Hashing Vector')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.tight_layout()
    plt.show()