import networkx as nx
import matplotlib.pyplot as plt

class TrieNode:
    """
    Trie Node class.
    Each node represents a character in the prefix.
    """
    def __init__(self):
        self.children = {}  # Dictionary to hold child nodes
        self.is_end = False  # Marks the end of a word

class Trie:
    """
    Trie (Prefix Tree) implementation for efficient string operations.

    Concepts:
    - A tree data structure where each node represents a prefix of stored strings.
    - Useful for autocomplete, spell checking, and prefix-based searches.

    Time Complexity:
    - Insert: O(m), where m is the length of the word
    - Search: O(m)
    - Starts with: O(m)

    Space Complexity:
    - O(total characters in all words), as each character is stored once

    ML/DL Implications:
    - Used in NLP for autocomplete systems, like in search engines or chatbots.
    - Efficient for handling large vocabularies in language models.

    Examples:
    - Insert "apple", "app", "application"
    - Search "app" -> True, "appl" -> False
    - Starts with "appl" -> True

    Edge Cases:
    - Empty string: Can be handled by setting root as end if needed
    - Duplicate inserts: No issue, as is_end is set
    - Non-existent prefixes: Returns False
    """
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end

    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

def build_trie_graph(trie):
    """
    Builds a NetworkX graph for Trie visualization.
    """
    G = nx.DiGraph()
    G.add_node("")  # Root node
    def dfs(node, path):
        for char, child in node.children.items():
            child_path = path + char
            G.add_edge(path, child_path)
            dfs(child, child_path)
    dfs(trie.root, "")
    return G

class SegmentTree:
    """
    Segment Tree for range queries (e.g., sum queries).

    Concepts:
    - A binary tree where each node represents a segment of the array.
    - Supports range queries and point updates efficiently.

    Time Complexity:
    - Build: O(n)
    - Update: O(log n)
    - Query: O(log n)

    Space Complexity:
    - O(n), as the tree has up to 4*n nodes

    ML/DL Implications:
    - Useful for efficient range queries in time series data, such as in RNNs or for feature extraction in ML pipelines.
    - Can be adapted for min/max queries in optimization problems.

    Examples:
    - Array: [1, 3, 5, 7, 9, 11]
    - Query sum from index 1 to 3: 3+5+7=15
    - Update index 1 to 10, then query: 10+5+7=22

    Edge Cases:
    - Single element array
    - Update out of bounds (not handled, assume valid indices)
    - Range queries with left > right (returns 0)
    """
    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)
        self.build(arr, 0, 0, self.n - 1)

    def build(self, arr, node, start, end):
        if start == end:
            self.tree[node] = arr[start]
            return
        mid = (start + end) // 2
        self.build(arr, 2 * node + 1, start, mid)
        self.build(arr, 2 * node + 2, mid + 1, end)
        self.tree[node] = self.tree[2 * node + 1] + self.tree[2 * node + 2]

    def update(self, idx, val, node=0, start=0, end=None):
        if end is None:
            end = self.n - 1
        if start == end:
            self.tree[node] = val
            return
        mid = (start + end) // 2
        if idx <= mid:
            self.update(idx, val, 2 * node + 1, start, mid)
        else:
            self.update(idx, val, 2 * node + 2, mid + 1, end)
        self.tree[node] = self.tree[2 * node + 1] + self.tree[2 * node + 2]

    def query(self, left, right, node=0, start=0, end=None):
        if end is None:
            end = self.n - 1
        if right < start or left > end:
            return 0
        if left <= start and end <= right:
            return self.tree[node]
        mid = (start + end) // 2
        return self.query(left, right, 2 * node + 1, start, mid) + self.query(left, right, 2 * node + 2, mid + 1, end)

def build_segment_tree_graph(segment_tree):
    """
    Builds a NetworkX graph for Segment Tree visualization.
    """
    G = nx.DiGraph()
    for i in range(len(segment_tree.tree)):
        if 2 * i + 1 < len(segment_tree.tree):
            G.add_edge(i, 2 * i + 1)
        if 2 * i + 2 < len(segment_tree.tree):
            G.add_edge(i, 2 * i + 2)
    return G

class FenwickTree:
    """
    Fenwick Tree (Binary Indexed Tree) for prefix sum queries.

    Concepts:
    - A data structure for efficient prefix sum calculations and updates.
    - Uses binary representation for indexing.

    Time Complexity:
    - Update: O(log n)
    - Query (prefix sum): O(log n)
    - Range query: O(log n)

    Space Complexity:
    - O(n)

    ML/DL Implications:
    - Useful for computing cumulative sums in feature engineering, e.g., in time series or sequence data.
    - Can be used in algorithms like dynamic programming with prefix sums.

    Examples:
    - Array: [1, 3, 5, 7, 9, 11]
    - Prefix sum up to index 3: 1+3+5+7=16
    - Range sum 1 to 3: 3+5+7=15

    Edge Cases:
    - Index 0: Query returns the first element
    - Update index 0
    - Size 1
    """
    def __init__(self, size):
        self.size = size
        self.tree = [0] * (size + 1)

    def update(self, idx, val):
        idx += 1
        while idx <= self.size:
            self.tree[idx] += val
            idx += idx & -idx

    def query(self, idx):
        idx += 1
        sum_val = 0
        while idx > 0:
            sum_val += self.tree[idx]
            idx -= idx & -idx
        return sum_val

    def range_query(self, left, right):
        return self.query(right) - (self.query(left - 1) if left > 0 else 0)

if __name__ == "__main__":
    # Trie demonstration
    print("Trie Demonstration:")
    trie = Trie()
    words = ["apple", "app", "application"]
    for word in words:
        trie.insert(word)
    print(f"Search 'app': {trie.search('app')}")  # True
    print(f"Starts with 'appl': {trie.starts_with('appl')}")  # True
    print(f"Search 'appl': {trie.search('appl')}")  # False

    # Visualize Trie
    G_trie = build_trie_graph(trie)
    plt.figure(figsize=(8, 6))
    nx.draw(G_trie, with_labels=True, node_color='lightblue', font_size=10)
    plt.title("Trie Visualization")
    plt.show()

    # Segment Tree demonstration
    print("\nSegment Tree Demonstration:")
    arr = [1, 3, 5, 7, 9, 11]
    st = SegmentTree(arr)
    print(f"Initial array: {arr}")
    print(f"Sum query 1 to 3: {st.query(1, 3)}")  # 15
    st.update(1, 10)
    print(f"After update index 1 to 10: {st.query(1, 3)}")  # 22

    # Visualize Segment Tree
    G_st = build_segment_tree_graph(st)
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G_st)
    nx.draw(G_st, pos, with_labels=True, labels={i: st.tree[i] for i in range(len(st.tree))}, node_color='lightgreen', font_size=8)
    plt.title("Segment Tree Visualization")
    plt.show()

    # Fenwick Tree demonstration
    print("\nFenwick Tree Demonstration:")
    ft = FenwickTree(len(arr))
    for i, val in enumerate(arr):
        ft.update(i, val)
    print(f"Range query 1 to 3: {ft.range_query(1, 3)}")  # 15
    ft.update(1, 10 - arr[1])  # Update difference
    print(f"After update index 1 to 10: {ft.range_query(1, 3)}")  # 22