import networkx as nx
import matplotlib.pyplot as plt

class UnionFind:
    """
    Union-Find (Disjoint Set Union) with path compression and union by rank.

    Concepts:
    - Data structure for managing disjoint sets, supporting union and find operations.
    - Used to track connected components.

    Time Complexity:
    - Find: Nearly O(1) amortized
    - Union: Nearly O(1) amortized

    Space Complexity:
    - O(n)

    ML/DL Implications:
    - Used in clustering algorithms, such as hierarchical clustering or connected components in image segmentation.
    - Efficient for graph algorithms in ML pipelines.

    Examples:
    - Union 0-1, 2-3, find 0 and 2: different sets
    - Union 1-2, now 0 and 2 same set

    Edge Cases:
    - Single element
    - Union same element
    - Find on invalid index (not handled)
    """
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px != py:
            if self.rank[px] > self.rank[py]:
                self.parent[py] = px
            elif self.rank[px] < self.rank[py]:
                self.parent[px] = py
            else:
                self.parent[py] = px
                self.rank[px] += 1

def build_union_find_graph(uf):
    """
    Builds a NetworkX graph for Union-Find visualization (showing the tree structure).
    """
    G = nx.DiGraph()
    for i in range(len(uf.parent)):
        if uf.parent[i] != i:
            G.add_edge(uf.parent[i], i)
    return G

def build_suffix_array(s):
    """
    Builds the suffix array for a string.

    Concepts:
    - An array of indices that represent the starting positions of the sorted suffixes of the string.
    - Useful for string processing tasks like pattern matching.

    Time Complexity:
    - O(n log n) using sorting

    Space Complexity:
    - O(n)

    ML/DL Implications:
    - Used in genomic sequence analysis for finding patterns in DNA/RNA sequences.
    - Efficient for substring searches in large texts.

    Examples:
    - String: "banana"
    - Suffixes: ["banana", "anana", "nana", "ana", "na", "a"]
    - Sorted: ["a", "ana", "anana", "banana", "na", "nana"]
    - Suffix Array: [5, 3, 1, 0, 4, 2]

    Edge Cases:
    - Empty string: []
    - Single character: [0]
    - All identical characters
    """
    suffixes = [(s[i:], i) for i in range(len(s))]
    suffixes.sort()
    return [idx for _, idx in suffixes]

def z_algorithm(s):
    """
    Computes the Z-array for a string.

    Concepts:
    - Z[i] is the length of the longest substring starting at i that matches the prefix of the string.
    - Efficient for string matching and pattern finding.

    Time Complexity:
    - O(n)

    Space Complexity:
    - O(n)

    ML/DL Implications:
    - Used in string matching algorithms in NLP, such as for tokenization or pattern recognition.
    - Helps in efficient substring searches.

    Examples:
    - String: "aaabaab"
    - Z-array: [0, 2, 1, 0, 2, 1, 0]

    Edge Cases:
    - Empty string: []
    - No matches: all 0 except z[0]=0
    - Entire string matches prefix
    """
    n = len(s)
    z = [0] * n
    l, r = 0, 0
    for i in range(1, n):
        if i < r:
            z[i] = min(r - i, z[i - l])
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1
        if i + z[i] > r:
            l, r = i, i + z[i]
    return z

if __name__ == "__main__":
    # Union-Find demonstration
    print("Union-Find Demonstration:")
    uf = UnionFind(5)
    uf.union(0, 1)
    uf.union(2, 3)
    print(f"Find 0: {uf.find(0)}, Find 2: {uf.find(2)}")  # 0, 2
    uf.union(1, 2)
    print(f"After union 1-2, Find 0: {uf.find(0)}, Find 2: {uf.find(2)}")  # 0, 0

    # Visualize Union-Find
    G_uf = build_union_find_graph(uf)
    plt.figure(figsize=(6, 4))
    nx.draw(G_uf, with_labels=True, node_color='lightcoral', font_size=10)
    plt.title("Union-Find Tree Visualization")
    plt.show()

    # Suffix Array demonstration
    print("\nSuffix Array Demonstration:")
    s = "banana"
    sa = build_suffix_array(s)
    print(f"String: {s}")
    print(f"Suffix Array: {sa}")
    # Print sorted suffixes
    for idx in sa:
        print(f"  {idx}: {s[idx:]}")

    # Visualize Suffix Array
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(sa)), sa, color='skyblue')
    plt.xticks(range(len(sa)), [f'{i}' for i in range(len(sa))])
    plt.xlabel("Suffix Array Index")
    plt.ylabel("Original Index")
    plt.title("Suffix Array Visualization")
    plt.show()

    # Z-algorithm demonstration
    print("\nZ-algorithm Demonstration:")
    s_z = "aaabaab"
    z_arr = z_algorithm(s_z)
    print(f"String: {s_z}")
    print(f"Z-array: {z_arr}")
    for i, z_val in enumerate(z_arr):
        if z_val > 0:
            print(f"  Position {i}: matches prefix of length {z_val} ('{s_z[:z_val]}')")