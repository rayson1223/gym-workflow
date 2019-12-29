import sys
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

matplotlib.style.use('ggplot')
import itertools
from mpl_toolkits.mplot3d import Axes3D


def main():
    # identified_ideal_result()
    X = [0, 1, 2] # cluster size
    Y = [1, 3, 5] # preference action
    Z = np.zeros(shape=(3,3))
    for xi, x in enumerate(X):
        for yi, y in enumerate(Y):
            Z[xi, yi] = x + y
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, vmin=0, vmax=7)
    ax.set_xlabel('Cluster Size')
    ax.set_ylabel('Cluster Number')
    ax.set_zlabel('Value')
    ax.set_title("test")
    ax.view_init(ax.elev, -120)
    fig.colorbar(surf)
    plt.show()


if __name__ == '__main__':
    sys.exit(main())