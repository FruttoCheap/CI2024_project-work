import random
import numpy as np
from src.node import OperatorNode, OperandNode, Node

# Constants for probabilities and ranges
CONSTANT_PROBABILITY = 0.3
OPERATOR_PROBABILITY = 0.6
OPERATOR_FIRST_PROBABILITY = 0.9
CONSTANT_RANGE = (-10, 10)

def generate_random_tree(max_depth, n_variables, current_depth=0, op=None):
    """Generate a random tree with a given maximum depth and number of variables."""
    def get_operator():
        """Generate an operator node with random children."""
        new_operator = random.choice(list(OperatorNode.OPERATORS.keys()))
        left = generate_random_tree(max_depth, n_variables, current_depth + 1, new_operator)
        right = None
        if new_operator in OperatorNode.BINARY_OPERATORS:
            right = generate_random_tree(max_depth, n_variables, current_depth + 1, new_operator)
        return OperatorNode(new_operator, left, right, current_depth)

    def get_operand():
        """Generate an operand node, either a constant or a variable."""
        if random.random() < CONSTANT_PROBABILITY:
            value = round(random.uniform(*CONSTANT_RANGE), 2)
            return OperandNode(value, current_depth)
        return OperandNode(f'x_{random.randint(0, n_variables-1)}', current_depth)

    if current_depth >= max_depth:
        return get_operand()
    if current_depth == 0 and random.random() < OPERATOR_FIRST_PROBABILITY:
        return get_operator()
    if random.random() < OPERATOR_PROBABILITY:
        return get_operator()
    return get_operand()

def initialize_population(pop_size, n_variables, max_depth=5):
    """Initialize a population of random trees."""
    return np.array([generate_random_tree(max_depth, n_variables) for _ in range(pop_size)])

def find_parent(root: Node, target: Node, parent: Node = None) -> Node | None:
    """Find the parent of a given node in the tree."""
    if root is target:
        return parent
    if isinstance(root, OperatorNode):
        if root.left:
            found = find_parent(root.left, target, root)
            if found:
                return found
        if root.right:
            found = find_parent(root.right, target, root)
            if found:
                return found
    return None

def replace_child(root: Node, old_child: Node, new_child: Node) -> None:
    """Replace a child node with a new node in the tree."""
    if not isinstance(root, OperatorNode):
        return
    if root.left is old_child:
        root.left = new_child
        return
    if root.right is old_child:
        root.right = new_child
        return
    if root.left:
        replace_child(root.left, old_child, new_child)
    if root.right:
        replace_child(root.right, old_child, new_child)

def simplify_expression(node: Node) -> Node:
    """Simplify an expression tree by reducing constant expressions."""
    if isinstance(node, OperandNode):
        return node
    if isinstance(node, OperatorNode):
        node.left = simplify_expression(node.left)
        if node.right:
            node.right = simplify_expression(node.right)
        if node.operator_symbol in OperatorNode.BINARY_OPERATORS:
            return simplify_binary_operator(node)
        return simplify_unary_operator(node)
    return node

def simplify_binary_operator(node: OperatorNode) -> Node:
    """Simplify binary operator nodes."""
    if isinstance(node.left, OperandNode) and isinstance(node.right, OperandNode):
        if isinstance(node.left.value, float) and isinstance(node.right.value, float):
            return OperandNode(node.function(node.left.value, node.right.value))
    if node.operator_symbol == '+':
        if isinstance(node.left, OperandNode) and node.left.value == 0:
            return node.right
        if isinstance(node.right, OperandNode) and node.right.value == 0:
            return node.left
    elif node.operator_symbol == '*':
        if isinstance(node.left, OperandNode) and node.left.value == 1:
            return node.right
        if isinstance(node.right, OperandNode) and node.right.value == 1:
            return node.left
        if isinstance(node.left, OperandNode) and node.left.value == 0:
            return OperandNode(0)
        if isinstance(node.right, OperandNode) and node.right.value == 0:
            return OperandNode(0)
    elif node.operator_symbol == '-':
        if isinstance(node.right, OperandNode) and node.right.value == 0:
            return node.left
        if isinstance(node.left, OperandNode) and isinstance(node.right, OperandNode) and node.left.value == node.right.value:
            return OperandNode(0)
    elif node.operator_symbol == '/':
        if isinstance(node.right, OperandNode) and node.right.value == 1:
            return node.left
        if isinstance(node.left, OperandNode) and isinstance(node.right, OperandNode) and node.left.value == node.right.value:
            return OperandNode(1)
    elif node.operator_symbol == '**':
        if isinstance(node.right, OperandNode) and node.right.value == 1:
            return node.left
        if isinstance(node.right, OperandNode) and node.right.value == 0:
            return OperandNode(1)
    return node

def simplify_unary_operator(node: OperatorNode) -> Node:
    """Simplify unary operator nodes."""
    if isinstance(node.left, OperandNode) and isinstance(node.left.value, float):
        return OperandNode(node.function(node.left.value))
    return node

def trim_population(population: list[Node]) -> list[Node]:
    """Trim the population to remove duplicate individuals."""
    unique_individuals = []
    seen_hashes = set()
    for ind in population:
        ind_hash = hash(str(ind))
        if ind_hash not in seen_hashes:
            seen_hashes.add(ind_hash)
            unique_individuals.append(ind)
    return unique_individuals