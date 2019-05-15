# coding: utf-8
# Author: Erick Armingol
# DFS with recursive function to solve a puzzle

from TreeStructures import Node

def recursive_DFS(solution, border_list, visited_nodes):
    node = None
    if len(border_list) > 0:
        node = border_list.pop()
        visited_nodes.append(node)
    else:
        return node
    if node.get_data() == solution:
        return node
    else:
        data = node.get_data()
        # Generate children by interchanging Left (L), Central (C) and Right (R) positions
        L_move = Node([data[1], data[0], data[2], data[3]], id = 'L')
        C_move = Node([data[0], data[2], data[1], data[3]], id = 'C')
        R_move = Node([data[0], data[1], data[3], data[2]], id = 'R')
        children = [L_move, C_move, R_move]
        node.set_children(children)
        for child in children:
            if not child.in_list(border_list) and not child.in_list(visited_nodes):
                border_list.append(child)
        node = recursive_DFS(solution, border_list, visited_nodes)
        return node


if __name__ == "__main__":
    initial_condition = [3, 1, 4, 2]
    solution = [1, 2, 3, 4]
    initial_node = Node(initial_condition, id = 'Initial-Node')
    border_list = [initial_node]

    solution_node = recursive_DFS(solution, border_list, [])

    if_child = True
    moves = [solution_node.get_id()]
    parent = solution_node.get_parent()
    while if_child:
        if parent != None:
            moves.append(parent.get_id())
            parent = parent.get_parent()
        else:
            if_child = False

    moves.reverse()
    print("Moves to get solution: ")
    for move in moves: print move + " -> ",
    print("Solution")

