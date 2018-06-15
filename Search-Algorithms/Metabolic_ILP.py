# coding: utf-8
# Author: Erick Armingol
# Integer Linear Programming with backtracking

import numpy as np

# Display settings for decimals
float_formatter = lambda x: "%.2f" % x # Display 2 decimals
np.set_printoptions(formatter={'float_kind':float_formatter}) #Set format of numpy-arrays


class Metabolic_ILP():
    def __init__(self, S, c, lb, ub, range_steps = 1.0):
        self.V = [0 for v in range(len(S[0]))]  # Rnxs velocities
        self.initial = [0 for v in range(len(S[0]))]
        self.S = S  ############ Add checker if len(S[0]) == len(V)
        self.c = c  ############ Add checker if len(c) == len(V)
        self.range_steps = range_steps
        self.variables_range = []
        count = 0
        for l, u in zip(lb, ub):
            if u >l:
                self.variables_range.append((l, u))
            elif u == l:
                self.initial[count] = u
                self.V[count] = u
                self.variables_range.append((l, u))
            else:
                print("Upper bound is lower than Lower bound for Position: ", count)
                print("Bounds were swapped to avoid errors")
                self.variables_range.append((u, l))
            count += 1
        #optimo = tuple(self.optimo)
        self.depth = 0
        self.solution_space = []
        self.x = self.backtracking(self.V[:], self.variables_range, self.initial, self.depth)
    
    def backtracking(self, variables, variables_range, optimum, depth):
        min = variables_range[depth][0]
        max = variables_range[depth][1]
        evaluation_list = np.arange(min, max + self.range_steps, self.range_steps).tolist()
        for v in evaluation_list:
            variables[depth] = v
            if depth < len(variables) - 1:
                if not self.if_completable(variables):  # not complateble if at least one constraint is not met
                    optimum = self.backtracking(variables[:], variables_range, optimum, depth + 1)
                else:   # if completable, it may be of the solution space
                    sol = self.evaluate_solution(variables)
                    check_visit = tuple(variables)
                    # if not visited, add to solution space
                    if not check_visit in self.solution_space:
                        opt_list = [variable for variable in variables]
                        solution = tuple(opt_list)
                        self.solution_space.append(solution)
                        # evaluate if this solution is better than previous optimum
                        if sol > self.evaluate_solution(optimum):
                            optimum = solution
                            optimum = self.backtracking(variables[:], variables_range, optimum, depth + 1)
            else:
                # we are in a leaf, check solution
                sol = self.evaluate_solution(variables)
                check_visit = tuple(variables)
                if self.if_completable(variables) and not check_visit in self.solution_space:
                    opt_list = [variable for variable in variables]
                    solution = tuple(opt_list)
                    self.solution_space.append(solution)
                    if sol > self.evaluate_solution(solution):
                        optimum = solution
        return optimum

    def evaluate_solution(self, variables):
        val = 0.0
        for i in range(len(variables)):
            val += self.c[i] * variables[i]
        return val

    def if_completable(self, variables):
        v_values = []
        for j in range(len(self.S)):  # For each metabolite
            aux_value = 0.0
            for i in range(len(variables)): # For rach rxn
                add = self.S[j][i] * variables[i]
                aux_value += add
            v_values.append(aux_value)
        if all(value == 0 for value in v_values):
            return True
        else:
            return False



if __name__ == "__main__":
    # Stoichiometric Matrix
    S = [[-1, -1, 0, 1, 0, 0],
         [1, 0, 1, 0, -1, 0],
         [0, 1, -1, 0, 0, -1]]

    # Objective function
    c = [5, 1, 0, 0, 0, 0]
    
    # Lower-bounds
    lb = [0, 0, 0, 4, 2, 2]
    
    # Upper-bounds
    ub = [10, 10, 10, 4, 2, 2]


    FBA = Metabolic_ILP(S, c, lb, ub, range_steps=0.1)
    print FBA.x
    print FBA.solution_space

    for vertex in FBA.solution_space:
        print vertex, " valor Z = ", FBA.evaluate_solution(vertex)