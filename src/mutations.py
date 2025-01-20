import random

from node import *
from utils import generate_random_tree, find_parent, replace_child

def mutate(individual: Node, max_depth: int, mutation_rate: float, n_variables: int) -> Node:
    """Mutate an individual with a given mutation rate."""
    if random.random() >= mutation_rate:
        return individual  # No mutation occurs

    # Clone the individual to avoid modifying the original
    individual = individual.clone()

    # Get all nodes in the tree
    nodes = get_all_nodes(individual)
    if not nodes:
        raise ValueError("Cannot mutate: The tree has no nodes.")

    # Select a random node for mutation
    mutate_node = random.choice(nodes)
    while mutate_node is None:
        mutate_node = random.choice(nodes)

    # Apply mutation strategies
    mutation_type = random.choice(["shrink", "subtree_replacement", "hoist", "tweak"])
    if mutation_type == "shrink":
        return apply_shrink_mutation(individual, mutate_node, n_variables)
    elif mutation_type == "subtree_replacement" and max_depth - mutate_node.depth > 1:
        new_subtree = generate_random_tree(max_depth, n_variables, mutate_node.depth)
        replace(individual, mutate_node, new_subtree)
    elif mutation_type == "hoist":
        return apply_hoist_mutation(individual, mutate_node)
    elif mutation_type == "tweak":
        tweak(mutate_node, n_variables)

    return individual

def apply_shrink_mutation(individual: Node, mutate_node: Node, n_variables: int) -> Node:
    """Apply shrink mutation by replacing a node with a terminal node."""
    # Find the parent of the mutate_node
    parent = find_parent(individual, mutate_node)

    # Generate a terminal node (variable or constant)
    terminal = OperandNode(
        value=random.choice([random.uniform(-10, 10), f"x_{random.randint(0, n_variables - 1)}"]),
        depth=mutate_node.depth  # Preserve depth for consistency
    )

    # Replace the mutate_node with the terminal node
    if parent:
        replace_child(parent, mutate_node, terminal)
    else:
        # If no parent, mutate_node is the root, replace the root directly
        individual = terminal

    return individual

def replace(individual, node: Node, new_subtree: Node):
    """Replace a node in the individual with a new subtree."""
    if node is None or new_subtree is None:
        raise ValueError("Cannot replace subtree: node is None.")

    parent = find_parent(individual, node)

    if parent is None:
        return

    if parent.left is node:
        parent.left = new_subtree.clone()
    elif parent.right is node:
        parent.right = new_subtree.clone()

def apply_hoist_mutation(individual: Node, mutate_node: Node) -> Node:
    """Apply hoist mutation by promoting a subtree."""
    # Ensure the mutate_node has at least one child to hoist
    if not isinstance(mutate_node, OperatorNode) or not mutate_node.left:
        return individual  # No hoist possible, return unchanged

    # Select the subtree to promote (left child by default)
    hoisted_subtree = mutate_node.left

    # Find the parent of the mutate_node
    parent = find_parent(individual, mutate_node)

    if parent:
        # Replace the mutate_node with the hoisted subtree in the parent's reference
        replace_child(parent, mutate_node, hoisted_subtree)
    else:
        # If the mutate_node is the root, replace the entire tree
        individual = hoisted_subtree

    return individual

def tweak(node: Node, n_variables: int):
    """Tweak a node by slightly modifying its value or operator."""
    if isinstance(node, OperatorNode):
        mutate_operator(node)
    elif isinstance(node, OperandNode):
        mutate_operand(node, n_variables)

def mutate_operator(node: OperatorNode):
    """Mutate an operator node by changing its operator."""
    if node.operator_symbol not in OperatorNode.BINARY_OPERATORS:
        # Unary operator
        available_operators = [op for op in OperatorNode.OPERATORS.keys() if op not in OperatorNode.BINARY_OPERATORS]
        available_operators.remove(node.operator_symbol)
    else:
        # Binary operator
        available_operators = [op for op in OperatorNode.BINARY_OPERATORS]
        available_operators.remove(node.operator_symbol)

    node.operator_symbol = random.choice(available_operators)
    node.function = OperatorNode.OPERATORS[node.operator_symbol]

def mutate_operand(node: OperandNode, n_variables: int):
    """Mutate an operand node by changing its value or variable index."""
    if isinstance(node.value, float):
        # Mutate constant value slightly
        node.value += random.uniform(-1, 1)
    elif isinstance(node.value, str) and node.value.startswith('x_'):
        # Change the variable index
        node.value = f"x_{random.randint(0, n_variables - 1)}"