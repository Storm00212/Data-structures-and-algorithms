import matplotlib.pyplot as plt

def compute_prefix_function(pattern):
    """
    Compute the prefix function (pi array) for KMP algorithm.

    Concept: The prefix function indicates the longest proper prefix that is also a suffix.
    Used in KMP to avoid unnecessary comparisons.
    """
    m = len(pattern)
    pi = [0] * m
    j = 0
    for i in range(1, m):
        while j > 0 and pattern[i] != pattern[j]:
            j = pi[j - 1]
        if pattern[i] == pattern[j]:
            j += 1
        pi[i] = j
    return pi

def kmp_search(text, pattern):
    """
    Knuth-Morris-Pratt (KMP) string matching algorithm.

    Concept: Uses a prefix function to skip characters, avoiding backtracking.
    Time Complexity: O(n + m), where n is text length, m is pattern length.
    Space Complexity: O(m) for the prefix array.
    ML/DL Implications: Efficient text search in NLP for pattern matching in large corpora, e.g., keyword extraction or plagiarism detection.
    Example Use Case: Searching for a substring in a document.
    Edge Cases: Empty text or pattern, pattern longer than text, no matches, multiple matches.
    """
    n, m = len(text), len(pattern)
    if m == 0:
        return []
    pi = compute_prefix_function(pattern)
    j = 0
    occurrences = []
    for i in range(n):
        while j > 0 and text[i] != pattern[j]:
            j = pi[j - 1]
        if text[i] == pattern[j]:
            j += 1
        if j == m:
            occurrences.append(i - m + 1)
            j = pi[j - 1]
    return occurrences

def rabin_karp(text, pattern, prime=101):
    """
    Rabin-Karp string matching algorithm using rolling hash.

    Concept: Computes hash values for substrings and compares them.
    Time Complexity: Worst case O((n-m+1)*m), but average O(n + m) with good hash.
    Space Complexity: O(1) extra space.
    ML/DL Implications: Fast approximate string matching in NLP, useful for fuzzy searches in text data.
    Example Use Case: Detecting similar patterns in genomic sequences or text documents.
    Edge Cases: Empty text or pattern, pattern longer than text, hash collisions (handled by string comparison).
    """
    n, m = len(text), len(pattern)
    if m == 0 or m > n:
        return []
    d = 256  # number of characters
    h = pow(d, m - 1, prime)
    p_hash = 0
    t_hash = 0
    for i in range(m):
        p_hash = (d * p_hash + ord(pattern[i])) % prime
        t_hash = (d * t_hash + ord(text[i])) % prime
    occurrences = []
    for i in range(n - m + 1):
        if p_hash == t_hash:
            if text[i:i + m] == pattern:
                occurrences.append(i)
        if i < n - m:
            t_hash = (d * (t_hash - ord(text[i]) * h) + ord(text[i + m])) % prime
            if t_hash < 0:
                t_hash += prime
    return occurrences

def visualize_prefix_function(pattern, pi):
    """
    Visualize the prefix function for KMP.
    """
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(pi)), pi, color='purple')
    plt.title(f'Prefix Function for Pattern: {pattern}')
    plt.xlabel('Position')
    plt.ylabel('Prefix Length')
    plt.xticks(range(len(pattern)), list(pattern))
    plt.show()

if __name__ == "__main__":
    text = "ABABDABACDABABCABAB"
    pattern = "ABABCABAB"

    # Test KMP
    kmp_results = kmp_search(text, pattern)
    print(f"KMP: Pattern '{pattern}' found at positions: {kmp_results}")

    # Test Rabin-Karp
    rk_results = rabin_karp(text, pattern)
    print(f"Rabin-Karp: Pattern '{pattern}' found at positions: {rk_results}")

    # Visualize prefix function
    pi = compute_prefix_function(pattern)
    print(f"Prefix function for '{pattern}': {pi}")
    visualize_prefix_function(pattern, pi)