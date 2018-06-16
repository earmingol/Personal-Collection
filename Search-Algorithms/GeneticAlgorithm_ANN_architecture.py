# coding: utf-8
# Author: Erick Armingol
# Genetic Algorithm to find the best ANN architecture

import numpy as np
import pandas as pd
import math
import random
import sklearn.datasets
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score


def initial_population(max_population, max_hidden_layers, max_neurons_per_layer):
    population = []
    for i in range(max_population):
        NN_arch = []
        for j in range(max_hidden_layers):
            if j == 0: # To avoid an architecture without neurons in the first hidden layer
                NN_arch.append(random.randint(1, max_neurons_per_layer))
            else:   # The followings layers may have no neurons
                NN_arch.append(random.randint(0, max_neurons_per_layer))
        population.append(NN_arch)
    return population

def evaluate_population_unit(individual, model, scoring, kfold):
    ''' Unit to perform parallel computing with pool.map'''
    # Generate architecture for individual
    try:
        H = tuple(individual[:individual.index(0)])
    except:
        H = tuple(individual)
    model.set_params(hidden_layer_sizes = H)
    scores = cross_val_score(model, data.data, data.target, scoring = scoring, cv = kfold, n_jobs = 1)
    return (scores.mean(), scores.std())

def evaluate_population(data, population, model, scoring = None, kfold = 5, n_jobs = 1):
    from contextlib import closing
    from multiprocessing import Pool, cpu_count
    from functools import partial

    if n_jobs < 0:
        agents = cpu_count() + 1 + n_jobs
        if agents < 0: agents = 1
    elif n_jobs > cpu_count():
        agents = cpu_count()
    elif n_jobs == 0:
        agents = 1
    else:
        agents = 1
    chunksize = 1
    fnc = partial(evaluate_population_unit, model = model, scoring = scoring, kfold = kfold)
    with closing(Pool(processes=agents)) as pool:
        parallel_evaluation = pool.map(fnc, population, chunksize)
    return parallel_evaluation

def select_parents(population, population_evaluation):
    evaluations = []
    neg = False
    if population_evaluation[0] < 0: # Check if scoring returns negative values
        neg = True
    for ev in population_evaluation:
        if neg: evaluations.append(abs(1.0/ev[0]))
        else: evaluations.append(ev[0])
    sorted_population = pd.Series(data=population, index=evaluations).sort_index().tolist()
    sorted_evaluations = sorted(evaluations, key=int)
    sum = abs(np.nansum(np.asarray(evaluations)))
    chr1_index = random.uniform(0, sum)
    chr2_index = random.uniform(0, sum)
    accum_sum = 0
    for i in range(len(sorted_evaluations)):
        accum_sum += abs(sorted_evaluations[i])
        if chr1_index < accum_sum:
            chromosome1 = sorted_population[i]
            break
    for i in range(len(sorted_evaluations)):
        accum_sum += abs(sorted_evaluations[i])
        if chr2_index < accum_sum:
            chromosome2 = sorted_population[i]
            break
    return chromosome1, chromosome2

def chromosomal_crossover(chro1, chro2, max_hidden_layers, max_neurons_per_layer):
    new_chromosomes = []
    try:
        tmp_chro1 = chro1[:chro1.index(0)]
    except:
        tmp_chro1 = chro1

    try:
        tmp_chro2 = chro2[:chro2.index(0)]
    except:
        tmp_chro2 = chro2

    # Parents with equal number of hidden layers
    if len(tmp_chro1) == len(tmp_chro2):
        # If number of hidden layers = 1 -> New chrom is the sum of the neurons of parents
        if len(tmp_chro1) == 1:
            new_chro = np.zeros(len(chro1))
            new_chro = new_chro.tolist()
            new_chro[0] = chro1[0] + chro2[0]
            if new_chro[0] > max_neurons_per_layer: # Not surpass the max number of neurons
                extra_neurons = new_chro[0] - max_neurons_per_layer
                new_chro[0] = max_neurons_per_layer
                try: new_chro[1] = extra_neurons
                except: new_chro.append(extra_neurons)
            new_chroms = [new_chro]
        # If number of hidden layers > 1 -> crossover
        else:
            cut_position = random.randint(1, len(tmp_chro1)-1)
            new_chro1 = tmp_chro1[:cut_position] + tmp_chro2[cut_position:]
            new_chro2 = tmp_chro2[:cut_position] + tmp_chro1[cut_position:]
            new_chroms = [new_chro1, new_chro2]
        while len(new_chroms[0]) < max_hidden_layers:
            for chrom in new_chroms: chrom.append(0)
    #Parents with diff numver of hidden layers
    else:
        tmp_chroms = [tmp_chro1, tmp_chro2]
        lens = np.asarray([len(tmp_chro1), len(tmp_chro2)])
        random_index = random.randrange(0, len(lens))   # Select new chroms size randomly
        min_chro = tmp_chroms.pop(int(np.argmin(lens)))
        max_chro = tmp_chroms.pop()
        cut_position = random.randint(0, len(min_chro)-1)
        new_chro1 = min_chro[:cut_position] + max_chro[cut_position:lens[random_index]]
        new_chro2 = max_chro[:cut_position] + min_chro[cut_position:]
        new_chroms = [new_chro1, new_chro2]
        while len(new_chroms[0]) < max_hidden_layers:
            new_chroms[0].append(0)
        while len(new_chroms[1]) < max_hidden_layers:
            new_chroms[1].append(0)
    new_chromosomes = new_chromosomes + new_chroms
    return new_chromosomes

def mutation(chromosomes, mutation_rate, max_neurons_per_layer):
    # Select one layer randomly and then add a random number of neurons between 1 and the number to complete the max_neurons_per_layer
    mutated_chromosomes = []
    for chro in chromosomes:
        p = random.random()
        if p < mutation_rate:
            try:
                tmp_chro = chro[:chro.index(0)]
            except:
                tmp_chro = chro
            p_chro = random.randint(0, len(tmp_chro)-1)
            value = tmp_chro[p_chro]
            r = random.random()
            if r > 0.5:
                diff = max_neurons_per_layer - value
                p_diff = random.randint(0, diff)
                tmp_chro[p_chro] += p_diff
            else:
                p_diff = random.randint(0, value-1)
                tmp_chro[p_chro] += -1 * p_diff
            while len(tmp_chro) < len(chro):
                tmp_chro.append(0)
            mutated_chromosomes.append(tmp_chro)
        else:
            mutated_chromosomes.append(chro)
    return mutated_chromosomes

def delete_worst_chromosomes(population, population_evaluation, number_of_new_chromosomes):
    evaluations = []
    for ev in population_evaluation:    # Check if scoring returns positive values
        evaluations.append(ev[0])
    sorted_population = pd.Series(data=population, index=evaluations).sort_index().tolist()
    sorted_evaluations = pd.Series(data=population_evaluation, index=evaluations).sort_index().tolist()
    count = 0
    while count < number_of_new_chromosomes:
        deleted = sorted_population.pop(0)
        deleted2 = sorted_evaluations.pop(0)
        count += 1
    return sorted_population, sorted_evaluations

def convergence(population_evaluation):
    mean = np.nanmean(np.asarray([ev[0] for ev in population_evaluation]))
    std = math.sqrt(np.nansum(np.asarray([ev[1]**2 for ev in population_evaluation])))
    return mean, std

def best_chromosome(population, population_evaluation):
    evaluations = []
    for ev in population_evaluation:    # Check if scoring returns positive values
        evaluations.append(ev[0])
    sorted_population = pd.Series(data=population, index=evaluations).sort_index().tolist()
    individual = sorted_population[-1]
    try:
        H = tuple(individual[:individual.index(0)])
    except:
        H = tuple(individual)
    return H


def genetic_algorithm_ANN(data,
                          model,
                          max_hidden_layers,
                          max_neurons_per_layer,
                          kfold = 5,
                          scoring = None,
                          max_generations = 50,
                          max_population = 10,
                          mutation_rate = 0.1,
                          coeff_variation_to_converge = 0.01,
                          n_jobs = 1):
    solved = False
    population = initial_population(max_population, max_hidden_layers, max_neurons_per_layer)
    generation = 0
    print "Calculating values for generation ", generation
    population_evaluation = evaluate_population(data, population, model, scoring, kfold, n_jobs)
    mean, std = convergence(population_evaluation)
    best = best_chromosome(population, population_evaluation)
    results = [(generation, mean, std, abs(std/mean), best)]
    while not solved:
        generation += 1
        print "Calculating values for generation ", generation
        for i in range(int(len(population)/2)):
            chromosome1, chromosome2 = select_parents(population, population_evaluation)
            new_chromosomes = chromosomal_crossover(chromosome1, chromosome2, max_hidden_layers, max_neurons_per_layer)
            mutated_new_chromosomes = mutation(new_chromosomes, mutation_rate, max_neurons_per_layer)
            new_evaluation = evaluate_population(data, mutated_new_chromosomes, model, scoring, kfold, n_jobs)
            population = population + mutated_new_chromosomes
            population_evaluation = population_evaluation + new_evaluation
            population, population_evaluation = delete_worst_chromosomes(population, population_evaluation, len(new_chromosomes))
        mean, std = convergence(population_evaluation)
        best = best_chromosome(population, population_evaluation)
        results.append((generation, mean, std, abs(std/mean), best))
        if generation >= max_generations:
            solved = True

        if abs(std/mean) <= coeff_variation_to_converge:
            solved = True
    labels = ['Generation', 'Mean', 'Std', 'CV', 'Best H']
    df = pd.DataFrame.from_records(results, columns = labels)
    print df
    return best

class Data():
    def __init__(self, X, y):
        self.data = X
        self.target = y

if __name__ == '__main__':
    data = sklearn.datasets.load_iris() # data have to be a class cointaining the inputs in a self.data and the outputs in self.target
    data = Data(data['data'], data['target'])

    model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(1,), random_state=1)

    max_hidden_layers = 4
    max_neurons_per_layer = 10
    kfold = 5
    scoring = 'accuracy'
    max_generations = 10
    max_population = 30
    mutation_rate = 0.1
    coeff_variation = 0.025
    H = genetic_algorithm_ANN(data,
                              model,
                              max_hidden_layers,
                              max_neurons_per_layer,
                              kfold,
                              scoring,
                              max_generations,
                              max_population,
                              mutation_rate,
                              coeff_variation,
                              n_jobs = -1)
