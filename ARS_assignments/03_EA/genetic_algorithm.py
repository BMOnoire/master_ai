import numpy as np
import random
import statistics as stats

X, Y, Z = 0, 1, 2

class GeneticAlgorithm:

    def __init__(self, function_name, mutation_prob, mutation_prob_step):
        self.fitness_function = function_name
        self.mutation_prob = mutation_prob
        self.mutation_prob_step = mutation_prob_step

    def initialize_population(self, pop_size, x_range, y_range):
        population = []
        for i in range(pop_size):
            x = random.choice(x_range)
            y = random.choice(y_range)
            z = self.fitness_function(x, y)
            population.append(np.array([x, y, z]))
        return np.array(population)

    def select_parents(self, parent_array, no_parents):  # truncated rank-based selection
        sorted_dataset = sorted(parent_array, key=lambda output: output[Z])
        return sorted_dataset[0:no_parents]

    def crossover_function(self, parent_array, pop_size, mantain):
        offspring = []
        parent_size = len(parent_array)
        offspring_size = pop_size
        offset = 0
        if mantain:
            offspring_size -= parent_size
        for i in range(offspring_size):
            if (i % parent_size) == 0:
                offset += 1
            parent_index = i % parent_size
            parent_index2 = (i+offset) % parent_size
            offspring_x = 0.5 * (parent_array[parent_index][X] + parent_array[parent_index2][X])
            offspring_y = 0.5 * (parent_array[parent_index][Y] + parent_array[parent_index2][Y])
            offspring_z = self.fitness_function(offspring_x, offspring_y)
            offspring.append([offspring_x, offspring_y, offspring_z])
        return offspring

    def mutation_function(self, offspring):
        for ofspr in offspring:
            if random.random() <= self.mutation_prob:
                ofspr[0] += np.random.uniform(-self.mutation_prob_step, self.mutation_prob_step)
            if random.random() <= self.mutation_prob:
                ofspr[1] += np.random.uniform(-self.mutation_prob_step, self.mutation_prob_step)
        return offspring

    def calculate_fitness(self, outputs):
        sorted_outputs = sorted(outputs, key=lambda output: output[Z])
        rank_list = []
        for out in sorted_outputs:
            rank_list.append(out[Z])
        return rank_list[0], stats.mean(rank_list), stats.stdev(rank_list)
