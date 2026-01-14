import matplotlib.pyplot as plt
import numpy as np

def edit_distance(s1, s2):
    """
    Computes the Edit Distance (Levenshtein Distance) using Dynamic Programming.

    Concept: Minimum number of operations (insert, delete, substitute) to transform s1 into s2.

    Time Complexity: O(m * n)
    Space Complexity: O(m * n)

    Applications: Text similarity in ML, spell checking, DNA sequence alignment in bioinformatics.

    Examples:
    - s1="kitten", s2="sitting"
      Distance: 3 (k->s, e->i, +g)

    Edge Cases:
    - Empty strings: distance = len(other)
    - Identical: 0
    - One char diff: 1
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n], dp

def visualize_edit_distance(dp, s1, s2):
    """Visualizes the DP table for Edit Distance as a heatmap."""
    plt.figure(figsize=(8, 6))
    plt.imshow(dp, cmap='coolwarm', origin='upper')
    plt.colorbar(label='Edit Distance')
    plt.title('Edit Distance DP Table Heatmap')
    plt.xlabel(f's2: {s2}')
    plt.ylabel(f's1: {s1}')
    plt.xticks(range(len(dp[0])), [''] + list(s2))
    plt.yticks(range(len(dp)), [''] + list(s1))
    plt.show()

def viterbi(observations, states, start_p, trans_p, emit_p):
    """
    Implements the Viterbi algorithm for Hidden Markov Models (HMM).

    Concept: Finds the most likely sequence of hidden states given observations.

    Time Complexity: O(T * N^2) where T is observations length, N is states
    Space Complexity: O(T * N)

    Applications: Speech recognition, POS tagging in NLP, bioinformatics.

    Example: Simple weather model with states ['Rainy', 'Sunny'], observations ['walk', 'shop', 'clean']

    Edge Cases: Single observation, single state.
    """
    T = len(observations)
    N = len(states)
    V = [{} for _ in range(T)]
    path = {}

    # Initialize
    for st in states:
        V[0][st] = start_p[st] * emit_p.get((st, observations[0]), 0)
        path[st] = [st]

    # Recursion
    for t in range(1, T):
        newpath = {}
        for st in states:
            (prob, prev_st) = max([(V[t-1][prev] * trans_p.get((prev, st), 0) * emit_p.get((st, observations[t]), 0), prev) for prev in states])
            V[t][st] = prob
            newpath[st] = path[prev_st] + [st]
        path = newpath

    # Termination
    (prob, st) = max([(V[T-1][s], s) for s in states])
    return prob, path[st], V

def visualize_viterbi(V, states, observations):
    """Visualizes the Viterbi DP table (probabilities) as a heatmap."""
    T = len(V)
    N = len(states)
    dp_matrix = [[V[t][st] for st in states] for t in range(T)]
    plt.figure(figsize=(8, 6))
    plt.imshow(dp_matrix, cmap='viridis', origin='upper')
    plt.colorbar(label='Probability')
    plt.title('Viterbi Algorithm DP Table Heatmap')
    plt.xlabel('Observations: ' + ', '.join(observations))
    plt.ylabel('States')
    plt.xticks(range(len(observations)), observations)
    plt.yticks(range(len(states)), states)
    plt.show()

def optimal_bst(keys, freq):
    """
    Computes the cost of Optimal Binary Search Tree using Dynamic Programming.

    Concept: Minimize expected search cost in BST with given keys and frequencies.

    Time Complexity: O(n^2)
    Space Complexity: O(n^2)

    Applications: Optimizing decision trees in ML, database indexing.

    Examples:
    - keys=['A','B','C','D'], freq=[3,3,1,1]
      Min cost: around 13

    Edge Cases:
    - Single key: freq[0]
    - Equal freq: balanced
    """
    n = len(keys)
    dp = [[0] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = freq[i]
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            for r in range(i, j + 1):
                left = dp[i][r - 1] if r > i else 0
                right = dp[r + 1][j] if r < j else 0
                cost = left + right + sum(freq[i:j + 1])
                if cost < dp[i][j]:
                    dp[i][j] = cost
    return dp[0][n - 1], dp

def visualize_optimal_bst(dp, keys):
    """Visualizes the DP table for Optimal BST as a heatmap."""
    plt.figure(figsize=(8, 6))
    plt.imshow(dp, cmap='plasma', origin='upper')
    plt.colorbar(label='Min Cost')
    plt.title('Optimal BST DP Table Heatmap')
    plt.xlabel('End Index')
    plt.ylabel('Start Index')
    plt.xticks(range(len(dp)), keys)
    plt.yticks(range(len(dp)), keys)
    plt.show()

if __name__ == "__main__":
    # Edit Distance Example
    s1 = "kitten"
    s2 = "sitting"
    dist, dp_edit = edit_distance(s1, s2)
    print(f"Edit Distance: {dist}")
    visualize_edit_distance(dp_edit, s1, s2)

    # Viterbi Example
    states = ['Rainy', 'Sunny']
    observations = ['walk', 'shop', 'clean']
    start_p = {'Rainy': 0.6, 'Sunny': 0.4}
    trans_p = {('Rainy', 'Rainy'): 0.7, ('Rainy', 'Sunny'): 0.3,
               ('Sunny', 'Rainy'): 0.4, ('Sunny', 'Sunny'): 0.6}
    emit_p = {('Rainy', 'walk'): 0.1, ('Rainy', 'shop'): 0.4, ('Rainy', 'clean'): 0.5,
              ('Sunny', 'walk'): 0.6, ('Sunny', 'shop'): 0.3, ('Sunny', 'clean'): 0.1}
    prob, path, V = viterbi(observations, states, start_p, trans_p, emit_p)
    print(f"Viterbi Prob: {prob}, Path: {path}")
    visualize_viterbi(V, states, observations)

    # Optimal BST Example
    keys = ['A', 'B', 'C', 'D']
    freq = [3, 3, 1, 1]
    cost, dp_bst = optimal_bst(keys, freq)
    print(f"Optimal BST Cost: {cost}")
    visualize_optimal_bst(dp_bst, keys)