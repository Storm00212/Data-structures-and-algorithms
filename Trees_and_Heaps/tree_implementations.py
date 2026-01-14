import networkx as nx
import matplotlib.pyplot as plt

class TreeNode:
    """
    Node class for binary trees.
    Each node has a value, left child, and right child.
    """
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BinaryTree:
    """
    Binary Tree class with traversal methods.
    A binary tree is a tree data structure where each node has at most two children.
    Time complexity for traversals: O(n) where n is the number of nodes.
    Space complexity: O(h) for recursive traversals, where h is the height of the tree.
    ML/DL implications: Trees are fundamental in decision tree algorithms, used for classification and regression.
    """
    def __init__(self, root=None):
        self.root = root

    def inorder_traversal(self, node, result=None):
        """
        Inorder traversal: Left, Root, Right.
        Useful for getting sorted order in BSTs.
        """
        if result is None:
            result = []
        if node:
            self.inorder_traversal(node.left, result)
            result.append(node.value)
            self.inorder_traversal(node.right, result)
        return result

    def preorder_traversal(self, node, result=None):
        """
        Preorder traversal: Root, Left, Right.
        Useful for creating a copy of the tree.
        """
        if result is None:
            result = []
        if node:
            result.append(node.value)
            self.preorder_traversal(node.left, result)
            self.preorder_traversal(node.right, result)
        return result

    def postorder_traversal(self, node, result=None):
        """
        Postorder traversal: Left, Right, Root.
        Useful for deleting the tree.
        """
        if result is None:
            result = []
        if node:
            self.postorder_traversal(node.left, result)
            self.postorder_traversal(node.right, result)
            result.append(node.value)
        return result

class BinarySearchTree(BinaryTree):
    """
    Binary Search Tree (BST) class.
    In BST, for each node, all elements in left subtree are less, right are greater.
    Time complexity: Average O(log n) for insert, delete, search; Worst case O(n) if unbalanced.
    Space complexity: O(n) for storing nodes.
    ML/DL implications: Used for efficient feature ranking, quick lookups in datasets.
    Example: Building BST from dataset features for fast median finding.
    Edge cases: Empty tree, single node, unbalanced tree (degenerates to linked list).
    """
    def __init__(self):
        super().__init__()

    def insert(self, value):
        """
        Insert a value into the BST.
        """
        if not self.root:
            self.root = TreeNode(value)
        else:
            self._insert_recursive(self.root, value)

    def _insert_recursive(self, node, value):
        if value < node.value:
            if node.left:
                self._insert_recursive(node.left, value)
            else:
                node.left = TreeNode(value)
        else:
            if node.right:
                self._insert_recursive(node.right, value)
            else:
                node.right = TreeNode(value)

    def search(self, value):
        """
        Search for a value in the BST.
        Returns True if found, False otherwise.
        """
        return self._search_recursive(self.root, value)

    def _search_recursive(self, node, value):
        if not node:
            return False
        if node.value == value:
            return True
        elif value < node.value:
            return self._search_recursive(node.left, value)
        else:
            return self._search_recursive(node.right, value)

    def delete(self, value):
        """
        Delete a value from the BST.
        """
        self.root = self._delete_recursive(self.root, value)

    def _delete_recursive(self, node, value):
        if not node:
            return node
        if value < node.value:
            node.left = self._delete_recursive(node.left, value)
        elif value > node.value:
            node.right = self._delete_recursive(node.right, value)
        else:
            # Node with one or no child
            if not node.left:
                return node.right
            elif not node.right:
                return node.left
            # Node with two children: get inorder successor
            temp = self._min_value_node(node.right)
            node.value = temp.value
            node.right = self._delete_recursive(node.right, temp.value)
        return node

    def _min_value_node(self, node):
        current = node
        while current.left:
            current = current.left
        return current

class AVLTree(BinarySearchTree):
    """
    AVL Tree: Self-balancing BST.
    Maintains balance factor |height(left) - height(right)| <= 1.
    Time complexity: O(log n) for all operations due to balancing.
    Space complexity: O(n).
    ML/DL implications: Ensures efficient operations in dynamic datasets, useful in real-time ML applications.
    Edge cases: Rotations for balance, empty tree.
    """
    def __init__(self):
        super().__init__()

    def get_height(self, node):
        if not node:
            return 0
        return max(self.get_height(node.left), self.get_height(node.right)) + 1

    def get_balance(self, node):
        if not node:
            return 0
        return self.get_height(node.left) - self.get_height(node.right)

    def right_rotate(self, y):
        x = y.left
        T2 = x.right
        x.right = y
        y.left = T2
        return x

    def left_rotate(self, x):
        y = x.right
        T2 = y.left
        y.left = x
        x.right = T2
        return y

    def insert(self, value):
        self.root = self._insert_avl(self.root, value)

    def _insert_avl(self, node, value):
        if not node:
            return TreeNode(value)
        if value < node.value:
            node.left = self._insert_avl(node.left, value)
        else:
            node.right = self._insert_avl(node.right, value)

        balance = self.get_balance(node)

        # Left Left
        if balance > 1 and value < node.left.value:
            return self.right_rotate(node)
        # Right Right
        if balance < -1 and value > node.right.value:
            return self.left_rotate(node)
        # Left Right
        if balance > 1 and value > node.left.value:
            node.left = self.left_rotate(node.left)
            return self.right_rotate(node)
        # Right Left
        if balance < -1 and value < node.right.value:
            node.right = self.right_rotate(node.right)
            return self.left_rotate(node)

        return node

    def delete(self, value):
        self.root = self._delete_avl(self.root, value)

    def _delete_avl(self, node, value):
        if not node:
            return node
        if value < node.value:
            node.left = self._delete_avl(node.left, value)
        elif value > node.value:
            node.right = self._delete_avl(node.right, value)
        else:
            if not node.left:
                return node.right
            elif not node.right:
                return node.left
            temp = self._min_value_node(node.right)
            node.value = temp.value
            node.right = self._delete_avl(node.right, temp.value)

        balance = self.get_balance(node)

        # Left Left
        if balance > 1 and self.get_balance(node.left) >= 0:
            return self.right_rotate(node)
        # Left Right
        if balance > 1 and self.get_balance(node.left) < 0:
            node.left = self.left_rotate(node.left)
            return self.right_rotate(node)
        # Right Right
        if balance < -1 and self.get_balance(node.right) <= 0:
            return self.left_rotate(node)
        # Right Left
        if balance < -1 and self.get_balance(node.right) > 0:
            node.right = self.right_rotate(node.right)
            return self.left_rotate(node)

        return node

def build_tree_from_list(nodes):
    """
    Helper to build a binary tree from a list for visualization.
    """
    if not nodes:
        return None
    root = TreeNode(nodes[0])
    queue = [root]
    i = 1
    while i < len(nodes):
        current = queue.pop(0)
        if i < len(nodes):
            current.left = TreeNode(nodes[i])
            queue.append(current.left)
            i += 1
        if i < len(nodes):
            current.right = TreeNode(nodes[i])
            queue.append(current.right)
            i += 1
    return root

def visualize_tree(root, title="Binary Tree"):
    """
    Visualize the tree using NetworkX.
    """
    if not root:
        print("Tree is empty.")
        return

    G = nx.DiGraph()
    def add_edges(node, parent=None):
        if node:
            G.add_node(node.value)
            if parent:
                G.add_edge(parent.value, node.value)
            add_edges(node.left, node)
            add_edges(node.right, node)

    add_edges(root)

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, arrows=False, node_color='lightblue', node_size=2000, font_size=16)
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    # Example: Binary Tree traversals
    bt = BinaryTree()
    bt.root = build_tree_from_list([1, 2, 3, 4, 5, 6, 7])
    print("Inorder:", bt.inorder_traversal(bt.root))
    print("Preorder:", bt.preorder_traversal(bt.root))
    print("Postorder:", bt.postorder_traversal(bt.root))
    visualize_tree(bt.root, "Binary Tree Example")

    # Example: BST operations
    bst = BinarySearchTree()
    features = [50, 30, 70, 20, 40, 60, 80]  # Simulating dataset features
    for f in features:
        bst.insert(f)
    print("BST Search 40:", bst.search(40))
    bst.delete(30)
    print("BST Inorder after delete 30:", bst.inorder_traversal(bst.root))
    visualize_tree(bst.root, "BST Example")

    # Example: AVL Tree
    avl = AVLTree()
    unbalanced_data = [10, 20, 30, 40, 50]  # Would be unbalanced in BST
    for d in unbalanced_data:
        avl.insert(d)
    print("AVL Inorder:", avl.inorder_traversal(avl.root))
    visualize_tree(avl.root, "AVL Tree Example")