# coding: utf-8
# PLE con backtracking
import numpy as np

# Display settings for decimals
float_formatter = lambda x: "%.2f" % x # Display 2 decimals
np.set_printoptions(formatter={'float_kind':float_formatter}) #Set to numpy arrays


class Metabolic_ILP():
    def __init__(self, S, c, lb, ub, range_steps = 1.0):
        self.V = [0 for v in range(len(S[0]))]  # Rnxs velocities
        self.inicial = [0 for v in range(len(S[0]))]
        self.S = S  ############ Add checker if len(S[0]) == len(V)
        self.c = c  ############ Add checker if len(c) == len(V)
        self.range_steps = range_steps
        self.rango_variables = []
        count = 0
        for l, u in zip(lb, ub):
            if u >l:
                self.rango_variables.append((l, u))
            elif u == l:
                self.inicial[count] = u
                self.V[count] = u
                self.rango_variables.append((l, u))
            else:
                print("Upper bound is lower than Lower bound for Position: ", count)
                print("Bounds were swapped to avoid errors")
                self.rango_variables.append((u, l))
            count += 1
        #optimo = tuple(self.optimo)
        self.profundidad = 0
        self.solution_space = []
        self.x = self.backtracking(self.V[:], self.rango_variables, self.inicial, self.profundidad)
    
    def backtracking(self, variables, rango_variables, optimo, profundidad):
        min = rango_variables[profundidad][0]
        max = rango_variables[profundidad][1]
        evaluation_list = np.arange(min, max + self.range_steps, self.range_steps).tolist()
        for v in evaluation_list:
            variables[profundidad] = v
            if profundidad < len(variables) - 1:
                if not self.es_completable(variables):  # Es no completable si incumple alguna restricción, probar otros valores para otras variables
                    optimo = self.backtracking(variables[:], rango_variables, optimo, profundidad + 1)
                else:   # Si cumple, es del espacio de soluciones
                    sol = self.evalua_solucion(variables)
                    check_visit = tuple(variables)
                    # Si no fue visitado, agregar a espacio de soluciones
                    if not check_visit in self.solution_space:
                        opt_list = [variable for variable in variables]
                        solution = tuple(opt_list)
                        self.solution_space.append(solution)
                        # Evaluar si mejora la solucion respecto al antiguo optimo
                        if sol > self.evalua_solucion(optimo):
                            optimo = solution
                            optimo = self.backtracking(variables[:], rango_variables, optimo, profundidad + 1)
            else:
                # estamos en una hoja, comprobamos solución
                sol = self.evalua_solucion(variables)
                check_visit = tuple(variables)
                if self.es_completable(variables) and not check_visit in self.solution_space:
                    opt_list = [variable for variable in variables]
                    solution = tuple(opt_list)
                    self.solution_space.append(solution)
                    if sol > self.evalua_solucion(solution):
                        optimo = solution
        return optimo

    def evalua_solucion(self, variables):
        val = 0.0
        for i in range(len(variables)):
            val += self.c[i] * variables[i]
        return val

    def es_completable(self, variables):
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
        print vertex, " valor Z = ", FBA.evalua_solucion(vertex)