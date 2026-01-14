"""
Graph Algorithms

This module implements key graph algorithms: DFS, BFS, Dijkstra's shortest path, and Kruskal's MST.
These algorithms are crucial in Machine Learning and Deep Learning:
- BFS/DFS: Graph traversal for exploring connected components, used in graph-based ML models like GNNs
- Dijkstra's: Shortest paths in weighted graphs, applications in recommendation systems (e.g., finding closest items)
- Kruskal's MST: Minimum spanning trees for clustering, network design, hierarchical clustering in ML

Time Complexities:
- DFS/BFS: O(V + E)
- Dijkstra's: O((V + E) log V) with binary heap
- Kruskal's: O(E log E) due to sorting edges

Space Complexities:
- DFS: O(V) for recursion stack
- BFS: O(V) for queue
- Dijkstra's: O(V) for distances + heap
- Kruskal's: O(V) for union-find

Edge Cases:
- Disconnected graphs: Algorithms may not visit all nodes
- Negative weights: Dijkstra's fails, use Bellman-Ford instead
- Empty graph: No operations
- Single node: Trivial cases

Examples:
- BFS: Level-order traversal in social networks
- Dijkstra's: Routing in transportation networks
- Kruskal's: Connecting cities with minimum cost
"""

import heapq
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt
from graph_representations import Graph

def dfs(graph, start, visited=None):
    """
    Depth-First Search traversal.

    Args:
        graph: Graph instance
        start: Starting vertex
        visited: Set of visited vertices

    Returns:
        List of visited vertices in DFS order
    """
    if visited is None:
        visited = set()
    visited.add(start)
    order = [start]
    if not graph.use_matrix:
        for neighbor, _ in graph.adj_list.get(start, []):
            if neighbor not in visited:
                order.extend(dfs(graph, neighbor, visited))
    else:
        # For matrix, iterate through vertices
        idx = graph.vertices.index(start)
        for j, weight in enumerate(graph.matrix[idx]):
            if weight != 0 and graph.vertices[j] not in visited:
                order.extend(dfs(graph, graph.vertices[j], visited))
    return order

def bfs(graph, start):
    """
    Breadth-First Search traversal.

    Args:
        graph: Graph instance
        start: Starting vertex

    Returns:
        List of visited vertices in BFS order
    """
    visited = set()
    queue = deque([start])
    visited.add(start)
    order = []
    while queue:
        vertex = queue.popleft()
        order.append(vertex)
        if not graph.use_matrix:
            for neighbor, _ in graph.adj_list.get(vertex, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        else:
            idx = graph.vertices.index(vertex)
            for j, weight in enumerate(graph.matrix[idx]):
                if weight != 0 and graph.vertices[j] not in visited:
                    visited.add(graph.vertices[j])
                    queue.append(graph.vertices[j])
    return order

def dijkstra(graph, start):
    """
    Dijkstra's algorithm for shortest paths from start vertex.

    Assumes non-negative weights.

    Args:
        graph: Graph instance
        start: Starting vertex

    Returns:
        Dict of distances and dict of previous vertices
    """
    distances = {v: float('inf') for v in (graph.adj_list if not graph.use_matrix else graph.vertices)}
    distances[start] = 0
    previous = {v: None for v in distances}
    pq = [(0, start)]
    while pq:
        dist, u = heapq.heappop(pq)
        if dist > distances[u]:
            continue
        if not graph.use_matrix:
            for v, weight in graph.adj_list.get(u, []):
                alt = dist + weight
                if alt < distances[v]:
                    distances[v] = alt
                    previous[v] = u
                    heapq.heappush(pq, (alt, v))
        else:
            idx = graph.vertices.index(u)
            for j, weight in enumerate(graph.matrix[idx]):
                if weight != 0:
                    v = graph.vertices[j]
                    alt = dist + weight
                    if alt < distances[v]:
                        distances[v] = alt
                        previous[v] = u
                        heapq.heappush(pq, (alt, v))
    return distances, previous

def kruskal(graph):
    """
    Kruskal's algorithm for Minimum Spanning Tree.

    Assumes undirected graph.

    Returns:
        List of edges in MST
    """
    if graph.directed or graph.use_matrix:
        raise ValueError("Kruskal's requires undirected graph with adjacency list")

    # Union-Find structure
    parent = {v: v for v in graph.adj_list}
    rank = {v: 0 for v in graph.adj_list}

    def find(v):
        if parent[v] != v:
            parent[v] = find(parent[v])
        return parent[v]

    def union(u, v):
        root_u = find(u)
        root_v = find(v)
        if root_u != root_v:
            if rank[root_u] > rank[root_v]:
                parent[root_v] = root_u
            elif rank[root_u] < rank[root_v]:
                parent[root_u] = root_v
            else:
                parent[root_v] = root_u
                rank[root_u] += 1

    # Get all edges
    edges = []
    for u in graph.adj_list:
        for v, w in graph.adj_list[u]:
            if u < v:  # Avoid duplicates in undirected
                edges.append((w, u, v))
    edges.sort()

    mst = []
    for w, u, v in edges:
        if find(u) != find(v):
            union(u, v)
            mst.append((u, v, w))
    return mst

def visualize_mst(original_graph, mst_edges):
    """
    Visualize the MST on the original graph.
    """
    nx_graph = original_graph.to_networkx()
    mst_graph = nx.Graph()
    for u, v, w in mst_edges:
        mst_graph.add_edge(u, v, weight=w)

    pos = nx.spring_layout(nx_graph)
    nx.draw(nx_graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', alpha=0.5)
    nx.draw(mst_graph, pos, with_labels=True, node_color='red', edge_color='red', width=2)
    plt.title("Original Graph (gray) and MST (red)")
    plt.show()

if __name__ == "__main__":
    # Create a sample graph
    g = Graph(directed=False, use_matrix=False)
    g.add_edge('A', 'B', 4)
    g.add_edge('A', 'C', 2)
    g.add_edge('B', 'C', 1)
    g.add_edge('B', 'D', 5)
    g.add_edge('C', 'D', 8)
    g.add_edge('C', 'E', 10)
    g.add_edge('D', 'E', 2)
    g.add_edge('D', 'F', 6)
    g.add_edge('E', 'F', 3)

    print("Graph adjacency list:", g.adj_list)

    # DFS
    dfs_order = dfs(g, 'A')
    print("DFS order from A:", dfs_order)

    # BFS
    bfs_order = bfs(g, 'A')
    print("BFS order from A:", bfs_order)

    # Dijkstra's
    distances, previous = dijkstra(g, 'A')
    print("Shortest distances from A:", distances)
    print("Previous vertices:", previous)

    # Kruskal's MST
    mst = kruskal(g)
    print("MST edges:", mst)

    # Visualize
    try:
        g.visualize()
        visualize_mst(g, mst)
    except ImportError:
        print("NetworkX or matplotlib not installed. Install with: pip install networkx matplotlib")

    # Edge case: Disconnected graph
    print("\nDisconnected graph:")
    disc_g = Graph()
    disc_g.add_edge('X', 'Y')
    disc_g.add_vertex('Z')
    print("DFS from X:", dfs(disc_g, 'X'))
    print("BFS from X:", bfs(disc_g, 'X'))