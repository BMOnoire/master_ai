import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def GetPlottingMatrices(x, y, func):
    X, Y = np.meshgrid(x, y)
    z = np.zeros(shape=(X.shape))
    for ix in range(X.shape[0]):
        for iy in range(X.shape[0]):
            z[ix, iy] = func(X[ix, iy], Y[ix, iy])
    return X, Y, z

def PlotFunction(x, y, func):
    X, Y, z = GetPlottingMatrices(x, y, func)
    fig = plt.figure()
    ax = fig.gca()
    plt.contourf(X, Y, z, cmap=cm.jet)
    plt.colorbar(aspect=5)
    plt.xlabel('x')
    plt.ylabel('y')
    return plt, ax

def DrawMarker(ax, x, y, value, is_final_mark):
    if is_final_mark == True:
        ax.plot(x, y, color='red', marker='x', markersize=20, markeredgewidth=4)
    else:
        ax.plot(x, y, color='magenta', marker='o', markersize=5, markeredgewidth=4)
    ax.annotate(value, (x, y))

def PlottingSegment(ax, Xa, Xb, Ya, Yb):  # segment from points A to B
    ax.plot([Xa, Xb], [Ya, Yb], linewidth=1, color='black')

def PlottingResults(values, x_list, y_list, function):
    v, ax = PlotFunction(x_list, y_list, function)
    for dot in values:
        DrawMarker(ax, dot[0], dot[1], "", False)
    v.show()

def plotting_performance(Z_list):
    best, mean, stdev = Z_list[0], Z_list[1], Z_list[2]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    # Set the axis lables
    ax.set_xlabel('Generations', fontsize=18)
    ax.set_ylabel('Fitness', fontsize=18)
    xaxis = np.array(range(0, len(Z_list[0])))

    # Line color for error bar
    color_best = 'red'
    color_mean = 'darkgreen'

    # Line style for each dataset
    lineStyle_best = {"linestyle": "-", "linewidth": 2, "markeredgewidth": 2, "elinewidth": 2, "capsize": 3}
    lineStyle_mean = {"linestyle": "-", "linewidth": 2, "markeredgewidth": 2, "elinewidth": 2, "capsize": 3}

    # Create an error bar for each dataset
    line_best = ax.errorbar(xaxis, best, **lineStyle_best, color=color_best, label='best')
    line_mean = ax.errorbar(xaxis, mean, yerr=stdev, **lineStyle_mean, color=color_mean, label='mean')

    # Label each dataset on the graph, xytext is the label's position
    #for i, txt in enumerate(best):
    #    ax.annotate(txt, xy=(xaxis[i], best[i]), xytext=(xaxis[i] + 0.03, best[i] + 0.3), color=color_best)

    #for i, txt in enumerate(mean):
    #    ax.annotate(txt, xy=(xaxis[i], mean[i]), xytext=(xaxis[i] + 0.03, mean[i] + 0.3), color=color_mean)

    # Draw a legend bar
    plt.legend(handles=[line_best, line_mean], loc='upper right')

    # Draw a grid for the graph
    ax.grid(color='lightgrey', linestyle='-')
    ax.set_facecolor('w')
    plt.savefig('Images\\performance.png')
    plt.show()

def plotting_diversity(diversity_array):


    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    # Set the axis lables
    ax.set_xlabel('Generations', fontsize=18)
    ax.set_ylabel('Diversity', fontsize=18)
    xaxis_diversity = np.array(range(0, len(diversity_array)))

    # Line color for error bar
    color_diversity = 'blue'

    # Line style for each dataset
    lineStyle_diversity = {"linestyle": "-", "linewidth": 2, "markeredgewidth": 2, "elinewidth": 2, "capsize": 3}

    # Create an error bar for each dataset
    line_diversity = ax.errorbar(xaxis_diversity, diversity_array, **lineStyle_diversity, color=color_diversity, label='diversity')

    # Draw a legend bar
    plt.legend(handles=[line_diversity], loc='upper right')

    # Draw a grid for the graph
    ax.grid(color='lightgrey', linestyle='-')
    ax.set_facecolor('w')
    plt.savefig('Images\\diversity.png')
    plt.show()

def plotting_test(f_map1, f_map2, f_map3, f_map4):

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    # Set the axis lables
    ax.set_xlabel('1, 10, 50 Generations', fontsize=18)
    ax.set_ylabel('Fitness', fontsize=18)
    xaxis = np.array([1,2,3])
    print(xaxis)
    # Line color for error bar
    color_map1 = 'blue'
    color_map2 = 'red'
    color_map3 = 'green'
    color_map4 = 'yellow'

    # Line style for each dataset
    lineStyle_map1 = {"linestyle": "-", "linewidth": 2, "markeredgewidth": 2, "elinewidth": 2, "capsize": 3}
    lineStyle_map2 = {"linestyle": "-", "linewidth": 2, "markeredgewidth": 2, "elinewidth": 2, "capsize": 3}
    lineStyle_map3 = {"linestyle": "-", "linewidth": 2, "markeredgewidth": 2, "elinewidth": 2, "capsize": 3}
    lineStyle_map4 = {"linestyle": "-", "linewidth": 2, "markeredgewidth": 2, "elinewidth": 2, "capsize": 3}

    # Create an error bar for each dataset
    line_map1 = ax.errorbar(xaxis, f_map1, **lineStyle_map1, color=color_map1, label='map1')
    line_map2 = ax.errorbar(xaxis, f_map2, **lineStyle_map2, color=color_map2, label='map2')
    line_map3 = ax.errorbar(xaxis, f_map3, **lineStyle_map3, color=color_map3, label='map3')
    line_map4 = ax.errorbar(xaxis, f_map4, **lineStyle_map4, color=color_map4, label='map4')


    # Draw a legend bar
    plt.legend(handles=[line_map1, line_map2, line_map3, line_map4], loc='upper right')

    # Draw a grid for the graph
    ax.grid(color='lightgrey', linestyle='-')
    ax.set_facecolor('w')
    plt.savefig('Images\\final_test.png')
    plt.show()