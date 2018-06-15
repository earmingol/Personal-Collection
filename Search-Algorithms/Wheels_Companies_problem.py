# coding: utf-8
import sys

import argparse

parser = argparse.ArgumentParser(description='Solution for optimizing the cost from selecting one company for each type of product. Each company could be picked once.')
parser.add_argument('--optimization', metavar="", dest="operation", default="min", help="'min' for Minimization and 'max' for Maximization (without quotes)")
args = parser.parse_args()

operation = args.operation

def backtracking(operation, variables, visited, optimum, data, levels, children, depth):
    for child in children:
        if child not in visited:
            variables[depth] = data[levels[depth]][child]
            visited2 = visited + [child]
            if depth < len(levels) - 1:
                optimum = backtracking(operation, variables[:], visited2, optimum, data, levels, children, depth + 1)
            else:
                sol = evaluate_solution(variables)
                if operation == 'max':
                    if sol > evaluate_solution(optimum[0]):
                        optimum[0] = variables
                        optimum[1] = visited2
                elif operation == 'min':
                    if sol < evaluate_solution(optimum[0]):
                        optimum[0] = variables
                        optimum[1] = visited2

    return optimum

def evaluate_solution(variables):
    fn = 0
    for var in variables:
        fn += var
    return fn

if __name__ == "__main__":
    type_of_wheels = ['T', 'H', 'V', 'W']
    companies = ['E1', 'E2', 'E3', 'E4']
    data = {'T' : {'E1' : 20, 'E2': 50, 'E3' : 60, 'E4' : 100},
            'H' : {'E1' : 30, 'E2': 50, 'E3' : 55, 'E4' : 80},
            'V' : {'E1' : 20, 'E2': 40, 'E3' : 50, 'E4' : 60},
            'W' : {'E1' : 40, 'E2': 50, 'E3' : 60, 'E4' : 70}}

    depth = 0
    visited = []
    variables = [0 for wheel in type_of_wheels]
    if operation == 'max':
        optimum = [[-1*sys.maxint for var in variables], []]
    elif operation == 'min':
        optimum = [[sys.maxint for var in variables], []]
    optimal_solution = backtracking(operation, variables, visited, optimum, data, type_of_wheels, companies, depth)
    print optimal_solution[0]
    print "Total: ", sum(optimal_solution[0])
    print optimal_solution[1]

