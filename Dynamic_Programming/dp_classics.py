import matplotlib.pyplot as plt
import numpy as np

def knapsack(weights, values, W):
    """
    Solves the 0/1 Knapsack problem using Dynamic Programming.

    Concept: Given weights and values of n items, and a knapsack capacity W,
    find the maximum value that can be obtained without exceeding the weight limit.
    Each item can be used at most once.

    Time Complexity: O(n * W)
    Space Complexity: O(n * W)

    ML/DL Implications: Used for resource allocation in ML training, such as selecting
    a subset of features or models within computational constraints.

    Examples:
    - weights = [1, 3, 4, 5], values = [1, 4, 5, 7], W = 7
      Max value: 9 (items 2 and 3: 4+5=9, weight 3+4=7)

    Edge Cases:
    - Empty list: return 0
    - W = 0: return 0
    - Single item: if weight <= W, return value, else 0
    """
    n = len(weights)
    if n == 0 or W == 0:
        return 0, [[0] * (W + 1) for _ in range(n + 1)]
    dp = [[0] * (W + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(W + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
            else:
                dp[i][w] = dp[i - 1][w]
    return dp[n][W], dp

def visualize_knapsack(dp):
    """Visualizes the DP table for 0/1 Knapsack as a heatmap."""
    plt.figure(figsize=(8, 6))
    plt.imshow(dp, cmap='viridis', origin='upper')
    plt.colorbar(label='Max Value')
    plt.title('0/1 Knapsack DP Table Heatmap')
    plt.xlabel('Weight Capacity')
    plt.ylabel('Number of Items')
    plt.xticks(range(len(dp[0])), range(len(dp[0])))
    plt.yticks(range(len(dp)), range(len(dp)))
    plt.show()

def lcs(X, Y):
    """
    Finds the length of the Longest Common Subsequence (LCS) using Dynamic Programming.

    Concept: LCS is the longest subsequence present in both sequences.
    A subsequence is derived from another sequence by deleting some or no elements
    without changing the order.

    Time Complexity: O(m * n)
    Space Complexity: O(m * n)

    ML/DL Implications: Used in sequence alignment in NLP, such as comparing DNA sequences
    or text similarity in machine translation.

    Examples:
    - X = "AGGTAB", Y = "GXTXAYB"
      LCS: "GTAB" (length 4)

    Edge Cases:
    - Empty strings: return 0
    - One empty: return 0
    - Identical strings: return length
    """
    m, n = len(X), len(Y)
    if m == 0 or n == 0:
        return 0, [[0] * (n + 1) for _ in range(m + 1)]
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i - 1] == Y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n], dp

def visualize_lcs(dp, X, Y):
    """Visualizes the DP table for LCS as a heatmap."""
    plt.figure(figsize=(8, 6))
    plt.imshow(dp, cmap='plasma', origin='upper')
    plt.colorbar(label='LCS Length')
    plt.title('LCS DP Table Heatmap')
    plt.xlabel(f'Y: {Y}')
    plt.ylabel(f'X: {X}')
    plt.xticks(range(len(dp[0])), [''] + list(Y))
    plt.yticks(range(len(dp)), [''] + list(X))
    plt.show()

def matrix_chain_order(p):
    """
    Solves the Matrix Chain Multiplication problem using Dynamic Programming.

    Concept: Given a sequence of matrices, find the most efficient way to multiply them.
    The problem is to find the optimal parenthesization to minimize scalar multiplications.

    Time Complexity: O(n^3)
    Space Complexity: O(n^2)

    ML/DL Implications: Relevant in optimizing computational graphs in DL frameworks,
    such as TensorFlow or PyTorch, for efficient matrix operations in neural networks.

    Examples:
    - p = [10, 20, 30, 40, 30]
      Min cost: 30000 (for matrices A10x20, B20x30, C30x40, D40x30)

    Edge Cases:
    - Single matrix: cost 0
    - Two matrices: p[0]*p[1]*p[2]
    """
    n = len(p) - 1
    if n == 0:
        return 0, []
    dp = [[0] * n for _ in range(n)]
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            for k in range(i, j):
                cost = dp[i][k] + dp[k + 1][j] + p[i] * p[k + 1] * p[j + 1]
                if cost < dp[i][j]:
                    dp[i][j] = cost
    return dp[0][n - 1], dp

def visualize_matrix_chain(dp):
    """Visualizes the DP table for Matrix Chain Multiplication as a heatmap."""
    plt.figure(figsize=(8, 6))
    plt.imshow(dp, cmap='inferno', origin='upper')
    plt.colorbar(label='Min Cost')
    plt.title('Matrix Chain Multiplication DP Table Heatmap')
    plt.xlabel('End Index')
    plt.ylabel('Start Index')
    plt.xticks(range(len(dp)), range(len(dp)))
    plt.yticks(range(len(dp)), range(len(dp)))
    plt.show()

if __name__ == "__main__":
    # 0/1 Knapsack Example
    weights = [1, 3, 4, 5]
    values = [1, 4, 5, 7]
    W = 7
    max_val, dp_knap = knapsack(weights, values, W)
    print(f"0/1 Knapsack Max Value: {max_val}")
    visualize_knapsack(dp_knap)

    # LCS Example
    X = "AGGTAB"
    Y = "GXTXAYB"
    lcs_len, dp_lcs = lcs(X, Y)
    print(f"LCS Length: {lcs_len}")
    visualize_lcs(dp_lcs, X, Y)

    # Matrix Chain Example
    p = [10, 20, 30, 40, 30]
    min_cost, dp_mat = matrix_chain_order(p)
    print(f"Matrix Chain Min Cost: {min_cost}")
    visualize_matrix_chain(dp_mat)