import operator
import numpy as np

class Node:
    def evaluate(self, x):
        """Evaluate the node with the given input."""
        raise NotImplementedError("Must implement evaluate method")

    def __str__(self):
        """Return the string representation of the node."""
        raise NotImplementedError("Must implement __str__ method")

    def clone(self):
        """Clone the node."""
        raise NotImplementedError("Must implement clone method")


class OperatorNode(Node):
    # Adding all NumPy mathematical functions to OPERATORS
    OPERATORS = {
        '+': operator.add,
        '-': operator.sub,
        '*': operator.mul,
        '/': lambda a, b: a / b if b != 0 else 1,  # Protected division
        'sin': lambda x: np.sin(np.clip(x, -1e10, 1e10)),  # Clamp input to avoid invalid values
        'cos': lambda x: np.cos(np.clip(x, -1e10, 1e10)),  # Clamp input to avoid invalid values
        'tan': lambda x: np.tan(np.clip(x, -1e10, 1e10)),  # Clamp input to avoid invalid values
        'log': lambda x: np.log(np.clip(x, 1e-10, None)),  # Avoid log(0) and negative values
        'exp': lambda x: np.exp(np.clip(x, -700, 700)),
        'sqrt': lambda x: np.sqrt(np.clip(x, 0, None)),  # Avoid sqrt of negative values
        'abs': lambda x: np.abs(x),
    }

    BINARY_OPERATORS = {'+', '-', '*', '/'}

    def __init__(self, operator_symbol, left=None, right=None, depth=0):
        self.operator_symbol = operator_symbol
        self.left = left
        self.right = right
        self.function = self.OPERATORS.get(operator_symbol)
        self.depth = depth

    def evaluate(self, x):
        """Evaluate the operator node with the given input."""
        left_val = self.left.evaluate(x)
        if self.operator_symbol not in self.BINARY_OPERATORS:
            result = self.function(left_val)
        else:
            right_val = self.right.evaluate(x)
            result = self.function(left_val, right_val)
        return result

    def __str__(self):
        """Return the string representation of the operator node."""
        if not self.right:
            return f"{self.operator_symbol}({self.left})"
        return f"({self.left} {self.operator_symbol} {self.right})"

    def clone(self):
        """Clone the operator node."""
        return OperatorNode(
            self.operator_symbol,
            self.left.clone() if self.left else None,
            self.right.clone() if self.right else None,
            self.depth,
        )

    def get_depth(self):
        """Get the depth of the operator node."""
        left_depth = self.left.get_depth() if self.left else 0
        right_depth = self.right.get_depth() if self.right else 0
        return max(left_depth, right_depth)


class OperandNode(Node):
    def __init__(self, value, depth=0):
        self.value = value  # Can be a constant or a variable
        self.depth = depth

    def evaluate(self, x):
        """Evaluate the operand node with the given input."""
        if isinstance(self.value, str):
            result = x[int(self.value.lstrip('x_'))]
        else:
            result = self.value
        return result

    def __str__(self):
        """Return the string representation of the operand node."""
        return str(self.value)

    def clone(self):
        """Clone the operand node."""
        return OperandNode(self.value, self.depth)

    def get_depth(self):
        """Get the depth of the operand node."""
        return self.depth + 1


def get_all_nodes(node):
    """Get all nodes in the tree."""
    nodes = [node]
    if isinstance(node, OperatorNode):
        nodes += get_all_nodes(node.left)
        nodes += get_all_nodes(node.right)
    return nodes