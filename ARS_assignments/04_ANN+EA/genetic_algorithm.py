import numpy as np
import random
import statistics as stats
import itertools
from copy import deepcopy

X, Y, Z = 0, 1, 2

class GeneticAlgorithm:

    def __init__(self, crossover_prob, mutation_prob, mutation_prob_step, ):
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.mutation_prob_step = mutation_prob_step

    def calculate_fitness(self, alpha, beta, array_A, array_B):
        fitness_array = []
        for value_A, value_B in zip(array_A, array_B):
            fitness = (alpha*value_A) + (beta*value_B)
            fitness_array.append(fitness)
        return fitness_array

    def order_population_and_fitness(self, pop_array, fitness_array):
        array, ordered_population, ordered_fitness = [], [], []

        for parent, fitness in zip(pop_array, fitness_array):
            array.append([parent, fitness])
        sorted_array = sorted(array, key=lambda array: array[1], reverse=True)

        for element in sorted_array:
            ordered_population.append(element[0])
            ordered_fitness.append(element[1])

        return ordered_population, ordered_fitness

    def select_parents(self, no_parents, parent_array):  # truncated rank-based selection
        #array = []
        #for parent, fitness in zip(parent_array, fitness_array):
        #    array.append([parent, fitness])
        #sorted_array = sorted(array, key=lambda array: array[1], reverse=True)
        truncated_array = parent_array[0:no_parents]
        # result, ordered_fitness = [], []
        # for element in sorted_array:
        #     result.append(element[0])
        #     ordered_fitness.append(element[1])
        return truncated_array

    def crossover_function(self, parent_array, pop_size, mantain):
        W0, W1 = 0, 1
        offspring = []
        parent_size = len(parent_array)
        offspring_size = pop_size
        offset = 0
        if mantain:
            offspring_size -= parent_size

        if offspring_size == 0: # no space for sons
            return parent_array

        for i in range(offspring_size):
            if (i % parent_size) == 0:
                offset += 1
            parent_index_A = i % parent_size
            parent_index_B = (i+offset) % parent_size

            #mix the weigths
            W0_A = parent_array[parent_index_A][W0]
            W1_A = parent_array[parent_index_A][W1]
            W0_B = parent_array[parent_index_B][W0]
            W1_B = parent_array[parent_index_B][W1]
            new_W0 = np.ones((len(W0_A), len(W0_A[0])))
            new_W1 = np.ones((len(W1_A), len(W1_A[0])))

            for i in range(len(W0_A)):
                for j in range(len(W0_A[0])):
                    if random.random() <= self.mutation_prob:
                        new_W0[i][j] = 0.5 * (W0_A[i][j] + W0_B[i][j])
                    else:
                        new_W0[i][j] = W0_A[i][j]

            for i in range(len(W1_A)):
                for j in range(len(W1_A[0])):
                    if random.random() <= self.mutation_prob:
                        new_W1[i][j] = 0.5 * (W1_A[i][j] + W1_B[i][j])
                    else:
                        new_W1[i][j] = W1_A[i][j]

            offspring.append([new_W0, new_W1])

        return offspring

    def mutation_function(self, offspring):
        for ofspr in offspring:
            if random.random() <= self.mutation_prob:
                for weight_matrix in ofspr:
                    for i in range(len(weight_matrix)):
                        for j in range(len(weight_matrix[0])):
                            if random.random() <= self.mutation_prob:
                                weight_matrix[i][j] += np.random.uniform(-self.mutation_prob_step, self.mutation_prob_step)
        return offspring

    def get_average_value(self, array):
        average_array = []
        for elements in array:
            average_array.append(stats.mean(elements))
        return average_array

    def get_normalized_value(self, array, top_range):
        normalized_array = []
        for value in array:
            normal = value/top_range
            normalized_array.append(normal)
        return normalized_array

    def calculate_diversity(self, population_in_all_epochs):
        diversity_array = []

        # get every generation
        for generation in population_in_all_epochs:
            total_diversity = 0

            #create an array of every combination of people
            tupled_generation = list(itertools.combinations(generation, 2))

            #for every tuple
            for tuple in tupled_generation:
                pop_A = deepcopy(tuple[0])
                pop_B = deepcopy(tuple[1])
                distance_matrix_W0 = np.absolute(np.subtract(pop_A[0], pop_B[0]))
                distance_matrix_W1 = np.absolute(np.subtract(pop_A[1], pop_B[1]))

                #sum every distance element
                sum_diversity_w0, sum_diversity_w1 = 0, 0
                for i in range(len(distance_matrix_W0)):
                    for j in range(len(distance_matrix_W0[0])):
                        sum_diversity_w0 += distance_matrix_W0[i][j]

                for i in range(len(distance_matrix_W1)):
                    for j in range(len(distance_matrix_W1[0])):
                        sum_diversity_w1 += distance_matrix_W1[i][j]

                total_diversity += (sum_diversity_w0 + sum_diversity_w1)

            diversity_array.append(total_diversity)

        return diversity_array

    #def calculate_fitness(self, alpha, beta, array_A, array_B):
    #    sorted_outputs = sorted(outputs, key=lambda output: output[Z])
    #    rank_list = []
    #    for out in sorted_outputs:
    #        rank_list.append(out[Z])
    #    return rank_list[0], stats.mean(rank_list), stats.stdev(rank_list)
