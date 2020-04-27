# -*- coding: utf-8 -*-

import numpy as np



    
# =============================================================================



def Plot2DDirect(x, y, func):
    plt, ax = Plot2D(x, y, func)
    plt.show()

def Plot3D(x, y, func):
    X, Y, z = GetPlottingMatrices(x, y, func)
    
    # === Plotting: 3D ===
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_surface(X, Y, z, cmap=cm.jet, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.75, aspect=5)
    plt.xlabel('x')
    plt.ylabel('y')
    
    return plt, ax

def Plot3DDirect(x, y, func):
    plt, ax = Plot3D(x, y, func)
    plt.show()
    
# =============================================================================
    
