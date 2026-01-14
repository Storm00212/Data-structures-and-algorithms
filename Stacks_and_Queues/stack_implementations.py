import matplotlib.pyplot as plt

class Node:
    """
    Node class for linked list implementation.
    Each node holds data and a reference to the next node.
    """
    def __init__(self, data):
        self.data = data
        self.next = None

class StackList:
    """
    Stack implementation using Python list.
    A stack is a LIFO (Last In First Out) data structure.
    Operations:
    - Push: O(1) amortized (list append)
    - Pop: O(1)
    - Peek: O(1)
    - Is_empty: O(1)
    - Size: O(1)
    Space: O(n)
    ML/DL implications: Stacks are used in parsing expressions in NLP models (e.g., syntax trees),
    managing function calls in recursive algorithms for backpropagation in neural networks.
    Examples: Evaluating postfix expressions in calculators, undo mechanisms in editors.
    Edge cases: Pop/peek on empty stack raises IndexError.
    """
    def __init__(self):
        self.stack = []

    def push(self, item):
        self.stack.append(item)

    def pop(self):
        if not self.is_empty():
            return self.stack.pop()
        raise IndexError("Pop from empty stack")

    def peek(self):
        if not self.is_empty():
            return self.stack[-1]
        raise IndexError("Peek from empty stack")

    def is_empty(self):
        return len(self.stack) == 0

    def size(self):
        return len(self.stack)

class StackLinkedList:
    """
    Stack implementation using linked list.
    Advantages over list: No resizing issues, constant time for push/pop.
    Operations:
    - Push: O(1)
    - Pop: O(1)
    - Peek: O(1)
    - Is_empty: O(1)
    - Size: O(1)
    Space: O(n)
    ML/DL implications: Useful in scenarios requiring dynamic memory, like parsing in compilers for ML code generation.
    Examples: Browser back button, expression evaluation.
    Edge cases: Pop/peek on empty stack raises IndexError.
    """
    def __init__(self):
        self.top = None
        self._size = 0

    def push(self, item):
        new_node = Node(item)
        new_node.next = self.top
        self.top = new_node
        self._size += 1

    def pop(self):
        if self.is_empty():
            raise IndexError("Pop from empty stack")
        item = self.top.data
        self.top = self.top.next
        self._size -= 1
        return item

    def peek(self):
        if self.is_empty():
            raise IndexError("Peek from empty stack")
        return self.top.data

    def is_empty(self):
        return self.top is None

    def size(self):
        return self._size

def infix_to_postfix(expression):
    """
    Converts infix expression to postfix using stack (Shunting-yard algorithm).
    Time: O(n)
    Space: O(n)
    ML/DL implications: Used in parsing mathematical expressions in data preprocessing for symbolic regression,
    or in NLP for parsing logical expressions in knowledge graphs.
    Examples: "A+B*C" -> "ABC*+"
    Edge cases: Invalid expressions (e.g., mismatched parentheses), empty expression.
    """
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}
    stack = StackList()
    output = []
    expression = expression.replace(" ", "")
    for char in expression:
        if char.isalnum():
            output.append(char)
        elif char == '(':
            stack.push(char)
        elif char == ')':
            while not stack.is_empty() and stack.peek() != '(':
                output.append(stack.pop())
            if not stack.is_empty():
                stack.pop()  # remove '('
        else:
            while (not stack.is_empty() and stack.peek() != '(' and
                   precedence.get(stack.peek(), 0) >= precedence.get(char, 0)):
                output.append(stack.pop())
            stack.push(char)
    while not stack.is_empty():
        output.append(stack.pop())
    return ''.join(output)

def next_greater_element(nums):
    """
    Finds next greater element for each element using monotonic stack.
    Monotonic stack maintains decreasing order.
    Time: O(n)
    Space: O(n)
    ML/DL implications: Used in time series analysis for peak detection, stock price predictions,
    or in computer vision for edge detection in monotonic sequences.
    Examples: [4,5,2,25] -> [5,25,25,-1]
    Edge cases: All elements decreasing (all -1), empty array ([] -> []), duplicates handled by first occurrence.
    """
    stack = []
    result = [-1] * len(nums)
    for i in range(len(nums)):
        while stack and nums[stack[-1]] < nums[i]:
            result[stack.pop()] = nums[i]
        stack.append(i)
    return result

def visualize_next_greater(nums):
    """
    Visualizes next greater elements using matplotlib bar chart with arrows.
    Shows the array and arrows pointing to next greater elements.
    """
    result = next_greater_element(nums)
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(nums)), nums, color='blue', alpha=0.7, label='Elements')
    for i in range(len(nums)):
        if result[i] != -1:
            # Find the index of the next greater element
            for j in range(len(nums)):
                if nums[j] == result[i]:
                    plt.arrow(i, nums[i], j - i, result[i] - nums[i] - 0.1, head_width=0.05, head_length=0.05, fc='red', ec='red')
                    break
    plt.xticks(range(len(nums)), [str(x) for x in nums])
    plt.title('Next Greater Element Visualization')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    print("Demonstrating Stack Implementations")
    print("=" * 40)

    print("\nStack using list:")
    s = StackList()
    s.push(1)
    s.push(2)
    s.push(3)
    print(f"Pop: {s.pop()}")
    print(f"Peek: {s.peek()}")
    print(f"Size: {s.size()}")

    print("\nStack using linked list:")
    sl = StackLinkedList()
    sl.push(1)
    sl.push(2)
    sl.push(3)
    print(f"Pop: {sl.pop()}")
    print(f"Peek: {sl.peek()}")
    print(f"Size: {sl.size()}")

    print("\nInfix to postfix conversion:")
    expr = "A+B*C"
    print(f"{expr} -> {infix_to_postfix(expr)}")

    print("\nNext greater element:")
    nums = [4, 5, 2, 25]
    print(f"{nums} -> {next_greater_element(nums)}")

    print("\nVisualizing next greater element...")
    visualize_next_greater(nums)