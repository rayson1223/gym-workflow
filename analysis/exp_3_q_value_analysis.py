import numpy as np
import csv
from collections import OrderedDict
import json
import matplotlib
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
import sys
import ast
import agents.utils.plotting as gym_plt

csv.field_size_limit(sys.maxsize)


def identified_ideal_result():
    makespan_record = {}
    with open('./data/all_cs_results.csv') as f:
        reader = csv.DictReader(f)
        for line in reader:
            degree = line['degrees']
            cs = line['cluster_size']
            if degree not in makespan_record:
                makespan_record[degree] = {}
            if cs not in makespan_record[degree]:
                makespan_record[degree][cs] = []
            makespan_record[degree][cs].append(line['exec_time'])

    def visualize(data):
        all_data = []
        all_median = []
        for cs, v in data.items():
            for i, d in enumerate(v):
                v[i] = int(d)
            all_data += v
            all_median.append(np.median(v))
        ti = []
        for i, v in enumerate(all_median):
            if v < np.percentile(all_data, 20):
                ti.append(i + 1)
        x = []
        for v in data.keys():
            x.append(int(v))
        plt.xlabel("Cluster Size")
        plt.ylabel("Makespan")
        # plt.ylim([0, 100])
        plt.axvspan(min(ti), max(ti), alpha=0.5)
        plt.boxplot(data.values())
        # plt.boxplot(data.values())
        plt.show()

    return visualize(makespan_record['0.5'])


def ideal_q_value(Q={}):
    X = range(1, 11, 1)
    Y = np.zeros(10)
    Y.fill(10)

    def line_version(X, Y):
        plt.xlabel("State (cluster number)")
        plt.ylabel("Q value (preference action)")
        plt.ylim([0, 11])
        plt.axhspan(8, 11, alpha=0.5)
        plt.plot(X, Y)
        plt.show()

    def mesh_version(Q, value=[0]):
        for k, v in Q.items():
            for i, m in enumerate(v):
                if i in value:
                    v[i] = 1
                else:
                    v[i] = 0
        gym_plt.plot_exp_2_action_value(Q)

    line_version(X, Y)
    mesh_version(Q, [9, 11])


def epi_analysis(data, episode=100):
    gym_plt.plot_exp_2_action_value(data, title="Experiment 3: Q-values overview - {} episodes".format(episode),
                                    show_analysis=False, opt_lower=8, opt_high=11, xlim=11, ylim=11)
    preference_action = []
    for k, v in data.items():
        preference_action.append(v.index(max(v)) + 1)
    X = [x + 1 for x in range(len(preference_action))]

    fig, ax = plt.subplots(figsize=(20, 10))

    plt.xlabel("State (cluster size)")
    plt.ylabel("Q value (preference action)")
    # plt.ylim([0, 10])
    ax.axhspan(8, 11, alpha=0.5)
    plt.ylim([0, 11])
    ax.plot(X, preference_action)
    plt.title("Q-value (preference action) over state")
    ax.set_xticks(range(0, 10))
    ax.set_yticks(range(0, 10))
    plt.show()


def plot_multi_lines(data):
    # plt.xlabel("State (cluster size)")
    # plt.ylabel("Q value (preference action)")
    # for k,v in data.items():
    #     plt.plot(v, label=k)
    # plt.legend()
    # plt.show()
    def polygon_under_graph(xlist, ylist):
        """
        Construct the vertex list which defines the polygon filling the space under
        the (xlist, ylist) line graph.  Assumes the xs are in ascending order.
        """
        return [(xlist[0], -4.), *zip(xlist, ylist), (xlist[-1], -4.)]

    fig = plt.figure(figsize=(30, 20))
    ax = fig.gca(projection='3d')

    # Make verts a list, verts[i] will be a list of (x,y) pairs defining polygon i
    verts = []

    # Set up the x sequence
    xs = list(data.keys())

    # The ith polygon will appear on the plane y = zs[i]
    zs = xs

    for i in zs:
        ys = data[i]
        verts.append(polygon_under_graph(xs, ys))

    poly = PolyCollection(verts, alpha=.1, offsets=True)
    colours = [colors.to_rgba(c) for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]
    poly.set_color(colours)
    # poly.set_facecolor(colours)
    ax.add_collection3d(poly, zs=zs, zdir='y')
    ax.autoscale_view(scaley=True, scalez=True)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xticks(range(0, 31))
    ax.set_yticks(range(0, 31))
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 31)
    ax.set_zlim(-4, 4)
    plt.show()


def plot_mesh(Q):
    if 'key' in Q.keys():
        del Q['key']
    min_x = min(k for k in Q.keys())
    max_x = max(k for k in Q.keys())
    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_x, max_x + 1)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros(shape=(max_x, max_x))
    for x, v in Q.items():
        for i, y in enumerate(v, start=0):
            Z[i, x-1] = y

    def plot_surface(X, Y, Z):
        def find_min_max_range(values):
            a = values.reshape(values.size)
            return min(a), max(a)

        minV, maxV = find_min_max_range(Z)
        fig = plt.figure(figsize=(20, 15))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap=matplotlib.cm.coolwarm, vmin=minV, vmax=maxV)
        ax.set_xticks(range(0, X.max()+1))
        ax.set_yticks(range(0, Y.max()+1))
        ax.set_xlabel('State (cluster number)')
        ax.set_ylabel('Available action (cluster number)')
        ax.set_zlabel('Q-Value (preference action)')
        ax.set_title("title")
        ax.view_init(ax.elev, -120)
        fig.colorbar(surf)
        # fig.savefig("./plots/{}.png".format("title"))
        # if show_plot:
        plt.show()

    plot_surface(X, Y, Z)


def main():
    # identified_ideal_result()
    q_record = {}
    with open('../agents/records/p2-exp1-epi-100-train-0-maintain-all-terminal-200-M10-B10.csv_episode_q_value.csv') as f:
        reader = csv.DictReader(f)
        for line in reader:
            fs = line['Q Value'].replace("array(", "").replace("])", "]").replace("\n", "")
            ans = ast.literal_eval(fs)
            # print(ans)
            q_record[int(line['episode'])] = ans
            # IT FUCKING WORKS
            # plt.plot_exp_3_action_value(ans)

    last_epi = len(q_record) - 1
    last_q = q_record[last_epi]

    sorted_q = OrderedDict(sorted(last_q.items()), key=lambda i: keyorder.index(i[0]))
    if 'key' in sorted_q:
        del sorted_q['key']

    # identified_ideal_result()
    # ideal_q_value(sorted_q)
    epi_analysis(sorted_q)
    # plot_multi_lines(sorted_q)
    # plot_mesh(sorted_q)


if __name__ == '__main__':
    sys.exit(main())
