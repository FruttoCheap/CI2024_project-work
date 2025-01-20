import random
from node import OperandNode, Node, get_all_nodes, OperatorNode
from utils import find_parent

def crossover(parent1: Node, parent2: Node) -> tuple[Node, Node]:
    # Clone the parents to avoid modifying the original trees
    parent1 = parent1.clone()
    parent2 = parent2.clone()

    # Get all nodes from both parents
    nodes1 = get_all_nodes(parent1)
    nodes2 = get_all_nodes(parent2)

    # Randomly select crossover points
    crossover_point1 = random.choice(nodes1)
    while crossover_point1 is None:
        crossover_point1 = random.choice(nodes1)

    crossover_point2 = random.choice(nodes2)
    while crossover_point2 is None:
        crossover_point2 = random.choice(nodes2)

    # Swap the subtrees if both crossover points are OperatorNodes
    if isinstance(crossover_point1, OperatorNode) and isinstance(crossover_point2, OperatorNode):
        swap_subtrees(crossover_point1, crossover_point2)
    elif isinstance(crossover_point1, OperandNode) and isinstance(crossover_point2, OperandNode):
        swap_operands(crossover_point1, crossover_point2)
    else:
        # Replace the subtree of one parent with the subtree of the other parent
        replace_subtree(parent1, parent2, crossover_point1, crossover_point2)

    return parent1, parent2

def swap_subtrees(node1: OperatorNode, node2: OperatorNode):
    # Swap the left subtrees
    node1.left, node2.left = node2.left.clone(), node1.left.clone()

    # Swap the right subtrees if they exist
    if node1.right is not None:
        if node2.right is not None:
            node1.right, node2.right = node2.right.clone(), node1.right.clone()
        else:
            node2.right = node1.right.clone()
            node1.right = None
    else:
        if node2.right is not None:
            node1.right = node2.right.clone()
            node2.right = None

    # Swap the operator symbols and functions
    node1.operator_symbol, node2.operator_symbol = node2.operator_symbol, node1.operator_symbol
    node1.function, node2.function = node2.function, node1.function

def swap_operands(node1: OperandNode, node2: OperandNode):
    # Swap the operand values
    node1.value, node2.value = node2.value, node1.value

def replace_subtree(ind1: Node, ind2: Node, subtree1: Node, subtree2: Node):
    if subtree1 is None or subtree2 is None:
        raise ValueError("Cannot replace subtree: node is None.")

    # Find the parents of the subtrees
    parent1 = find_parent(ind1, subtree1)
    parent2 = find_parent(ind2, subtree2)

    if parent1 is None or parent2 is None:
        return

    # Replace the subtrees in the parents
    if parent1.left is subtree1 and parent2.left is subtree2:
        parent1.left = subtree2.clone()
        parent2.left = subtree1.clone()
    elif parent1.right is subtree1 and parent2.right is subtree2:
        parent1.right = subtree2.clone()
        parent2.right = subtree1.clone()
    elif parent1.left is subtree1 and parent2.right is subtree2:
        parent1.left = subtree2.clone()
        parent2.right = subtree1.clone()
    elif parent1.right is subtree1 and parent2.left is subtree2:
        parent1.right = subtree2.clone()
        parent2.left = subtree1.clone()