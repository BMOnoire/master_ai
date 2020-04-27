# -*- coding: utf-8 -*-
import genetic_algorithm as ga
import plotting as plot
import numpy as np
from math import cos, pi
from copy import deepcopy

# ===  Defining Benchmark Functions  ===
def Rosenbrock(x, y):  # MIN 0
    a, b = 0, 100;
    return (a - x) ** 2 + b * (y - x ** 2) ** 2

def Rastrigin(x, y):  # MIN 0
    return 10 * 2 + (x ** 2 - 10 * cos(2 * pi * x)) + (y ** 2 - 10 * cos(2 * pi * y))


# ===  INPUTS  ===
FITNESS_FUNCTION = Rastrigin  # Rosenbrock
POPULATION_SIZE = 60
PARENTS_NUMBER = int(POPULATION_SIZE / 4)
MUTATION_PROBABILITY = 0.05
MUTATION_P_STEP = 1.5
MANTAIN_PARENTS = True

GENETIC_EPOCHS = 50
INDIVIDUAL_STEPS = 5


# ===  Calculating x,y Values  ===
LIM_m_x, LIM_M_x = -1, 1
LIM_m_y, LIM_M_y = -1, 1
if FITNESS_FUNCTION.__name__ == "Rosenbrock":
    LIM_m_x, LIM_M_x = -2, 2
    LIM_m_y, LIM_M_y = -1, 3
else:  # Rastrigin
    LIM_m_x, LIM_M_x = -5, 5
    LIM_m_y, LIM_M_y = -5, 5
x_range_list = np.linspace(LIM_m_x, LIM_M_x, 200)
y_range_list = np.linspace(LIM_m_y, LIM_M_y, 200)


# ===  MAIN  ===
Pi, Po, Z = 0, 1, 2
X, Y = 0, 1

# INITS
geneticAlgorithm = ga.GeneticAlgorithm(FITNESS_FUNCTION, MUTATION_PROBABILITY, MUTATION_P_STEP)
FF_results = [[], [], []]  # best, media,stdev


# INIT DATASET
dataset = geneticAlgorithm.initialize_population(POPULATION_SIZE, x_range_list, y_range_list)
plot.PlottingResults(dataset, x_range_list, y_range_list, FITNESS_FUNCTION)

best, media, stdev = geneticAlgorithm.calculate_fitness(dataset)
FF_results[0].append(best)
FF_results[1].append(media)
FF_results[2].append(stdev)

copied_dataset = np.array(deepcopy(dataset))
for i in range(1, GENETIC_EPOCHS + 1):
    parents = geneticAlgorithm.select_parents(copied_dataset, PARENTS_NUMBER)
    copied_dataset = geneticAlgorithm.crossover_function(parents, POPULATION_SIZE, MANTAIN_PARENTS)
    copied_dataset = geneticAlgorithm.mutation_function(copied_dataset)
    if MANTAIN_PARENTS:
        all_individuals= parents + copied_dataset
        copied_dataset = deepcopy(all_individuals)

    best, media, stdev = geneticAlgorithm.calculate_fitness(copied_dataset)
    FF_results[0].append(best)
    FF_results[1].append(media)
    FF_results[2].append(stdev)

    if i % 10 == 0:
        plot.PlottingResults(copied_dataset, x_range_list, y_range_list, FITNESS_FUNCTION)

#plot.PlottingResults(copied_dataset, x_range_list, y_range_list, FITNESS_FUNCTION)
plot.PlottingPerformance(FF_results)
