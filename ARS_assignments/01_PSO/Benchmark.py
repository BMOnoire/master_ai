import random
import math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm


# import GD as gradient


# Benchmark functions and plots


class Benchmark:

    def rosenbrock(self, x, y):
        a = 0.5
        b = 2
        return (a - x) ** 2 + b * (y - x ** 2) ** 2

    def rastrigin(self, x, y):
        n = 10
        return (x ** 2 - 10 * np.cos(2 * np.pi * x)) + (y ** 2 - 10 * np.cos(2 * np.pi * y)) + 20

    def plot_func(self, fun):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if fun == self.rosenbrock:
            x = np.linspace(-5, 5, 100)
            y = np.linspace(0, 10, 100)
        else:
            x = y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)
        zs = np.array([fun(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])
        Z = zs.reshape(X.shape)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.rainbow, antialiased=True)
        plt.show()

    def connectpoints(self, x, y, i, j):
        x1 = x[i]
        x2 = x[j]
        y1 = y[i]
        y2 = y[j]
        plt.plot([x1, x2], [y1, y2], linestyle='dashed', color="red")

    def plot_func_particles(self, fun, a, b):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if fun == self.rosenbrock:
            x = y = np.linspace(-5, 5, 100)
        else:  # self.rastrigin
            x = y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)
        zs = np.array([fun(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])  # {(x1,y1),..,,(x, yn)}
        Z = zs.reshape(X.shape)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.rainbow, antialiased=True)

        # Printing particles
        for i in np.arange(0, len(a)):
            A, B = np.meshgrid(a[i], b[i])
            zab = np.array([fun(a[i], b[i]) for a[i], b[i] in zip(np.ravel(A), np.ravel(B))])  # ask
            Z = zs.reshape(X.shape)
            ax.scatter(a[i], b[i], zab, color='black', marker="*", s=30)
            # if i == 0:
            #     ax.scatter(a[i], b[i], zab, color='black', marker="s", s=30)
            # else:
            #     ax.scatter(a[i], b[i], zab, color='black', marker="*", s=10)
                #self.connectpoints(a, b, i-1, i)
        plt.show()

    def plot_func_particles_history(self, fun, a, b):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if fun == self.rosenbrock:
            x = y = np.linspace(-5, 5, 100)
        else:  # self.rastrigin
            x = y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)
        zs = np.array([fun(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])  # {(x1,y1),..,,(x, yn)}
        Z = zs.reshape(X.shape)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.rainbow, antialiased=True)

        # Printing particles
        for i in np.arange(0, len(a)):
            A, B = np.meshgrid(a[i], b[i])
            zab = np.array([fun(a[i], b[i]) for a[i], b[i] in zip(np.ravel(A), np.ravel(B))])  # ask
            Z = zs.reshape(X.shape)
            if i == 0:
                ax.scatter(a[i], b[i], zab, color='black', marker="s", s=30)
            else:
                ax.scatter(a[i], b[i], zab, color='black', marker="*", s=10)
                self.connectpoints(a, b, i - 1, i)

        plt.show()

    def plot_func_2D(self, fun):
        fig = plt.figure()
        ax = fig.gca()
        # x = y = np.linspace(-5, 5, 100)
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-1, 9, 100)
        X, Y = np.meshgrid(x, y)
        zs = np.array([fun(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])
        Z = zs.reshape(X.shape)
        cfset = ax.contourf(X, Y, Z, cmap='coolwarm')
        fig.colorbar(cfset, shrink=0.5, aspect=5)
        plt.show()


class Particle:
    def __init__(self, dimensions, min_vel, max_vel, func_to_optimize):
        self.func_to_optimize = func_to_optimize
        self.history_y, self.history_x = np.empty(0), np.empty(0)
        self.position = np.asarray(
            [random.uniform(-5, +5) for i in range(dimensions)])  # need to do a array of positions
        self.pbest = self.position  # Best previous position
        self.gbest = self.position  # Best position among neighbours
        self.min_vel = min_vel
        self.max_vel = max_vel
        self.velocity = np.asarray([1] * dimensions)  # Min speed at the beginning [1,1]

    def update_vel(self, a_decay):
        # Best configuration for a, b & c given in the slides
        b = 2
        c = 2
        a = 0.9 - a_decay
        print(a);
        # Updating with current vel, previous best location & best neighbour location
        self.velocity = a * self.velocity + b * random.random() * (self.pbest - self.position) + c * random.random() * (
                self.gbest - self.position)
        # Check it does not exceed maximum speed & does exceed the minimum
        np.where(self.velocity > self.max_vel, self.max_vel, self.velocity)
        np.where(self.velocity < -self.max_vel, - self.max_vel, self.velocity)

    def update_pos(self):
        # Assuming that Euler Integratoin is 1
        self.position = self.position + self.velocity
        if self.func_to_optimize.__name__ == 'rosenbrock':
            self.position = np.clip(self.position, -5, 5)
        else:  # self.rastrigin
            self.position = np.clip(self.position, -5, 5)  ########## to review

    def calc_performance(self):
        performance = self.func_to_optimize(self.position[0],
                                            self.position[1])  # Value of the function at the current location
        if performance < self.func_to_optimize(self.pbest[0],
                                               self.pbest[1]):  # Update best position if it's better than previous ones
            # print("\n\nOld: ", self.func_to_optimize(self.pbest[0], self.pbest[1]), " -- New: ", performance)
            self.pbest = self.position
        return performance

    def set_gbest(self, gbest):
        self.gbest = gbest
        return self.gbest

    def get_position(self):
        return self.position

    def save_history(self):
        self.history_x = np.append(self.history_x, self.position[0])
        self.history_y = np.append(self.history_y, self.position[1])
        print("\n\nhistory ", self.history_x, self.history_y)


# Works for variable number of dimensions (variables)
# but the visualization only works for two
class PSO:
    def __init__(self, n_particles, dimensions, range_neighbours, func_to_optimize, min_vel=0,
                 max_vel=999999):  # (5,2,20,rastrigin)
        self.particles = [Particle(dimensions, min_vel, max_vel, func_to_optimize) for i in range(n_particles)]
        self.range_neighbours = range_neighbours
        self.func_to_optimize = func_to_optimize

    # Now it works with all the neighbours!!
    # -> Has to be changed to the closest ones (distance function)
    def calc_and_set_gbest(self):
        # HERE the distance function and etc would start
        for part1 in self.particles:
            local_best = +99999999
            n = 0
            for part2 in self.particles:  # Iterates between all the possible neighbours
                if self.distance(part1, part2) < self.range_neighbours:  # If it's inside the neighbour range
                    if part2.calc_performance() < local_best:  # Value is better than the other neighbours
                        gbest = part2.position
                        local_best = part2.calc_performance()
                        n += 1
                # print("\nI have",n,"neighbours")

    def fit(self, max_iterations):
        for i in range(max_iterations):
            # Updating location of the best particle in the neighbourhood
            self.calc_and_set_gbest()
            # Decay parameter 'a' in the particles -> -0.5 in 1000 generations
            decay = i * 0.0005

            # Updating position & velocity

            for particle in self.particles:
                particle.update_vel(decay)
                particle.update_pos()
                if i == 0:
                    particle.save_history()
                elif i % 250 == 0:
                    particle.save_history()

            # Maybe we can add an exit condition (ex: particles barely move)
            # condition
            # if condition:
            #	break

    def distance(self, part1, part2):
        summ = 0
        for i in range(len(part1.position)):
            summ += abs(part1.position[i] - part2.position[i])
        return summ

    def visualization(self):
        if len(self.particles[0].position) != 2:
            print("\n >> The visualization is only possible with 2 dimensions")
        else:
            x = [particle.position[0] for particle in self.particles]  # X coordinates
            y = [particle.position[1] for particle in self.particles]  # Y coordinates
            print("position ", x, y)
            bm = Benchmark()
            bm.plot_func_particles(self.func_to_optimize, x, y)
            # for particle1 in self.particles:
            #     bm.plot_func_particles(self.func_to_optimize, x, y)

    def visualization_history(self):
        if len(self.particles[0].position) != 2:
            print("\n >> The visualization is only possible with 2 dimensions")
        else:
            bm = Benchmark()
            for particle1 in self.particles:
                bm.plot_func_particles_history(self.func_to_optimize, particle1.history_x, particle1.history_y)


# class GD:
#
#     def __init__(self, dimensions, func_to_optimize, gamma=0.01, precision=0.00001):
#         self.func_to_optimize = func_to_optimize
#         self.position = np.asarray([random.uniform(-5, +5) for i in range(dimensions)])
#         print("initial value ", self.position)
#         self.gamma = gamma
#         self.precision = precision
#
#     def execute(self, max_iterations):
#         for i in range(max_iterations):
#             # pos_x= np.append(pos_x, self.position[0])
#             current_x = self.position[0]
#             self.position[0] = current_x - self.gamma * self.func_to_optimize(current_x, self.position[1])
#             step = self.position[0] - current_x
#             if abs(step) <= self.precision:
#                 break
#
#         print("Minimum at ", self.position[0], self.position[1])
#     # pos_y= np.append(pos_y, self.position[1])
#     # def visualization():
#     #     pass


class GD2:

    def __init__(self, dimensions, func_to_optimize, gamma=0.01, precision=0.01):
        self.func_to_optimize = func_to_optimize
        self.position = np.asarray([random.uniform(-5, +5) for i in range(dimensions)])
        print("initial value ", self.position)
        self.gamma = gamma
        self.precision = np.float64(precision)

    def execute(self, max_iterations):
        for i in range(max_iterations):
            # pos_x= np.append(pos_x, self.position[0])
            current_x = self.position
            if abs(current_x[0]) < self.precision or abs(current_x[1]) < self.precision:
                print("BREAK")
                break
            self.position = current_x - self.gamma * self.func_to_optimize(current_x[0], current_x[1])
            step = self.position - current_x
            print(type(self.position[0]), type(self.precision))
            print(self.position)

        print("Minimum at ", self.position)
        return self.position
    # pos_y= np.append(pos_y, self.position[1])
    # def visualization():
    #     pass


############# MAIN EXECUTION #############

### Limits of the space


bm = Benchmark()

# bm.plot_func(bm.rastrigin)

# x = [0, 0, 2]
# y = [0, 6, 4]

# bm.plot_func_particles(bm.rosenbrock, x, y)

pt = Particle(2, 0, 9999, bm.rosenbrock)

pt.update_vel(0.01)
pt.update_pos()
a = pt.calc_performance()
x, y = pt.get_position()
print("this is", x, y)

# bm.plot_func_particles(bm.rosenbrock, [x], [y])

pso = PSO(20, 2, 20, bm.rosenbrock)

# gd = GD(2, bm.rastrigin)
# gd.execute(1000)


pso.visualization()  # SHOWS THE FUNCTION WITH THE PARTICLES
pso.fit(1000)  # ITERATES THE NUMBER OF ITERATIONS
pso.visualization()
# pso.visualization_history()
# pso.fit(10000)
# pso.visualization()
# pso.fit(10)
# pso.visualization()
# pso.fit(10)
# pso.visualization()

print("\n\n")

# gd = GD2(2, bm.rosenbrock)
# gd.execute(100)


# bm.plot_func(bm.rosenbrock)
