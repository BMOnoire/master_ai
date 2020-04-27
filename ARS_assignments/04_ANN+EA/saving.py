import os
import numpy as np

def create_directory(dir_name):  # function to create directories required if they don't exist
    try:
        os.mkdir(dir_name)
    except OSError:
        print("Creation of the directory %s failed" % dir_name)
    else:
        print("Successfully created the directory %s " % dir_name)


# Saves the model in a txt file
def save_model_weight(epoch, pop, weights1, weights2):
    name = "gen" + str(epoch) + " " + str(pop)
    np.savetxt(("Save\\" + name + "-w1.txt"), weights1, fmt="%s")
    np.savetxt(("Save\\" + name + "-w2.txt"), weights2, fmt="%s")




# Saves the score in a txt file
def save_model_score(epoch, population_size, score_array, collision_array, score_average, collision_average, score_norm, collision_norm, fitness):
    scores = []
    for i in range(population_size):
        string = \
            "Robot: " + str(i) \
            + " | Score: " + str(score_array[i]) \
            + " | Collision avoided: " + str(collision_array[i]) \
            + " | Score averages: [" + str(score_average[i]) \
            + "] | Collision avoided averages: [" + str(collision_average[i]) \
            + "] | Score normalized: [" + str(score_norm[i]) \
            + "] | Collision avoided normalized: [" + str(collision_norm[i]) \
            + "] | Fitness function: [" + str(fitness[i]) \
            + "] \n"
        scores.append(string)

    np.savetxt(("Score\\" + "gen" + str(epoch) + "-score.txt"), scores, fmt="%s")


# Load a model from a txt file
def load_model(epoch, pop):
    try:
        name = "gen" + str(epoch) + " " + str(pop)
        w1 = np.loadtxt((("Save\\" + name + "-w1.txt")))
        w2 = np.loadtxt(("Save\\" + name + "-w2.txt"))
    except:
        print("The indicated model doesn't exist!")
        exit(1)
    return w1, w2

def save_model_weight_training(epoch, weights1, weights2):
    name = "gen" + str(epoch)+ " best"
    np.savetxt(("Best\\" + name + "-w1.txt"), weights1, fmt="%s")
    np.savetxt(("Best\\" + name + "-w2.txt"), weights2, fmt="%s")

def load_model_training(epoch):
    try:
        name = "gen" + str(epoch) + " best"
        print(name)
        w1 = np.loadtxt((("Best\\" + name + "-w1.txt")))
        w2 = np.loadtxt(("Best\\" + name + "-w2.txt"))
    except:
        print("The indicated model doesn't exist!")
        exit(1)
    return w1, w2