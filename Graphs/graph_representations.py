"""
Graph Representations and Basic Operations

This module implements graph data structures with adjacency list and adjacency matrix representations.
Graphs are fundamental in Machine Learning and Deep Learning for modeling relationships, such as:
- Neural network architectures (e.g., computational graphs in TensorFlow/PyTorch)
- Social network analysis for clustering and recommendation systems
- Knowledge graphs for NLP tasks
- Graph Neural Networks (GNNs) for node/edge classification

Time Complexities:
- Adjacency List: Add vertex O(1), Add edge O(1), Remove edge O(degree)
- Adjacency Matrix: Add vertex O(V), Add edge O(1), Remove edge O(1)

Space Complexities:
- Adjacency List: O(V + E)
- Adjacency Matrix: O(V^2)

Edge Cases:
- Empty graph: No vertices or edges
- Disconnected components: Multiple isolated subgraphs
- Self-loops: Edges from vertex to itself (allowed in some representations)
- Parallel edges: Multiple edges between same vertices (adj list supports, matrix doesn't easily)

Example: Representing a small social network
Vertices: People, Edges: Friendships
"""

import networkx as nx
import matplotlib.pyplot as plt

class Graph:
    """
    Graph class supporting directed/undirected graphs with adjacency list or matrix representation.
    """

    def __init__(self, directed=False, use_matrix=False):
        """
        Initialize graph.

        Args:
            directed (bool): True for directed graph, False for undirected
            use_matrix (bool): True for adjacency matrix, False for adjacency list
        """
        self.directed = directed
        self.use_matrix = use_matrix
        if use_matrix:
            self.matrix = []  # 2D list for adjacency matrix
            self.vertices = []  # List of vertex identifiers
        else:
            self.adj_list = {}  # Dict: vertex -> list of (neighbor, weight)

    def add_vertex(self, v):
        """
        Add a vertex to the graph.

        Time: O(1) for adj list, O(V) for matrix (resizing)
        """
        if self.use_matrix:
            if v not in self.vertices:
                self.vertices.append(v)
                # Expand matrix: add column to existing rows
                for row in self.matrix:
                    row.append(0)
                # Add new row
                self.matrix.append([0] * len(self.vertices))
        else:
            if v not in self.adj_list:
                self.adj_list[v] = []

    def add_edge(self, u, v, weight=1):
        """
        Add an edge between u and v with optional weight.

        Time: O(1) for both representations
        """
        self.add_vertex(u)
        self.add_vertex(v)
        if self.use_matrix:
            i = self.vertices.index(u)
            j = self.vertices.index(v)
            self.matrix[i][j] = weight
            if not self.directed:
                self.matrix[j][i] = weight
        else:
            self.adj_list[u].append((v, weight))
            if not self.directed:
                self.adj_list[v].append((u, weight))

    def remove_edge(self, u, v):
        """
        Remove edge between u and v.

        Time: O(degree) for adj list, O(1) for matrix
        """
        if self.use_matrix:
            if u in self.vertices and v in self.vertices:
                i = self.vertices.index(u)
                j = self.vertices.index(v)
                self.matrix[i][j] = 0
                if not self.directed:
                    self.matrix[j][i] = 0
        else:
            if u in self.adj_list:
                self.adj_list[u] = [edge for edge in self.adj_list[u] if edge[0] != v]
            if not self.directed and v in self.adj_list:
                self.adj_list[v] = [edge for edge in self.adj_list[v] if edge[0] != u]

    def to_networkx(self):
        """
        Convert to NetworkX graph for visualization.
        """
        if self.directed:
            nx_graph = nx.DiGraph()
        else:
            nx_graph = nx.Graph()

        if self.use_matrix:
            for i, u in enumerate(self.vertices):
                for j, v in enumerate(self.vertices):
                    if self.matrix[i][j] != 0:
                        nx_graph.add_edge(u, v, weight=self.matrix[i][j])
        else:
            for u in self.adj_list:
                for v, w in self.adj_list[u]:
                    nx_graph.add_edge(u, v, weight=w)
        return nx_graph

    def visualize(self):
        """
        Visualize the graph using NetworkX and matplotlib.
        """
        nx_graph = self.to_networkx()
        pos = nx.spring_layout(nx_graph)
        nx.draw(nx_graph, pos, with_labels=True, node_color='lightblue', edge_color='gray')
        labels = nx.get_edge_attributes(nx_graph, 'weight')
        nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=labels)
        plt.title("Graph Visualization")
        plt.show()

if __name__ == "__main__":
    # Example: Small social network (undirected, adj list)
    print("Creating undirected graph with adjacency list...")
    g = Graph(directed=False, use_matrix=False)
    g.add_edge('Alice', 'Bob')
    g.add_edge('Alice', 'Charlie')
    g.add_edge('Bob', 'David')
    g.add_edge('Charlie', 'David')

    print("Adjacency List:", g.adj_list)

    # Visualize
    try:
        g.visualize()
    except ImportError:
        print("NetworkX or matplotlib not installed. Install with: pip install networkx matplotlib")

    # Example: Directed graph with matrix
    print("\nCreating directed graph with adjacency matrix...")
    g2 = Graph(directed=True, use_matrix=True)
    g2.add_edge('A', 'B', 2)
    g2.add_edge('B', 'C', 3)
    g2.add_edge('A', 'C', 1)

    print("Vertices:", g2.vertices)
    print("Matrix:")
    for row in g2.matrix:
        print(row)

    # Edge case: Empty graph
    print("\nEmpty graph:")
    empty_g = Graph()
    print("Adj list:", empty_g.adj_list)

    # Disconnected components
    print("\nGraph with disconnected components:")
    disc_g = Graph()
    disc_g.add_edge('X', 'Y')
    disc_g.add_vertex('Z')  # Isolated vertex
    print("Adj list:", disc_g.adj_list)