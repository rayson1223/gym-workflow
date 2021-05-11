import numpy as np
import csv
from collections import OrderedDict
import json
import matplotlib.pyplot as plt
import math
import sys
import ast
import agents.utils.plotting as gym_plt

csv.field_size_limit(sys.maxsize)


def plot_ideal_reward():
    x = np.arange(-10., 10., 0.2)

    def sigmoid(x):
        a = []
        for item in x:
            a.append(1 / (1 + math.exp(-item)))
        return a

    sig = sigmoid(x)
    plt.plot(x, sig)
    plt.tick_params(axis='both', which='both', labelbottom='off',
                    labelleft='off')
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title("Ideal Episode Reward over Episode(s)")
    plt.show()


def plot_ideal_total_reward():
    x = np.arange(2.5, 5, 0.1)

    def log(x):
        a = []
        for item in x:
            a.append(math.cos(item))
        return a

    sig = log(x)
    plt.plot(x, sig)
    plt.tick_params(axis='both', which='both', labelbottom='off',
                    labelleft='off')
    plt.xlabel("Episode")
    plt.ylabel("Episode Total Reward")
    plt.title("Ideal Episode Total Reward over Episode(s)")
    plt.show()


def plot_ideal_epi_length():
    t1 = np.arange(0.0, 5.0, 0.01)

    def f(t):
        return np.exp(-t) * np.cos(2 * np.pi * t)

    plt.plot(t1, f(t1))
    plt.tick_params(axis='both', which='both', labelbottom='off',
                    labelleft='off')
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Ideal Episode Length over Episode(s)")
    plt.show()


def plot_ideal_q_value(q, epi):
    for k, v in q.items():
        # (cs, cn)
        if k < 15:
            q[k] = [0, 1, 0]
        elif k == 15:
            q[k] = [1, 0, 0]
        else:
            q[k] = [0, 0, 1]
    gym_plt.v1_plot_action_value(q,
                                 title="Experiment 1: Ideal Action Values Overview - {} episodes".format(epi + 1))


def plot_idea_cluster(xlabel='Clustering Parameter'):
    x = np.arange(0, 21, 1)

    def polar(x):
        a = []
        for item in x:
            a.append(math.pow(0.2 * (item - 10), 2))
        return a

    y = polar(x)
    plt.plot(x, y)
    plt.tick_params(axis='both', which='both', labelbottom=False,
                    labelleft=False)
    plt.xlabel(xlabel)
    plt.ylabel("Makespan (s)")
    plt.title("Makespan (s) over clustering parameter")
    plt.show()


def main():
    q_record = {}
    with open('../agents/records/exp1-cn-training-0-epi-100-vm-10.csv_episode_q_value.csv') as f:
        reader = csv.DictReader(f)
        for line in reader:
            fs = line['Q Value'].replace("array(", "").replace("])", "]").replace("\n", "")
            # for num in range(10):
            #     fs = fs.replace("{}:".format(num+1), "\"{}\":".format(num+1))
            ans = ast.literal_eval(fs)
            # print(ans)
            q_record[int(line['episode'])] = ans
            # IT FUCKING WORKS
            # gym_plt.v1_plot_action_value(ans)

    last_epi = len(q_record) - 1
    last_q = q_record[last_epi]

    sorted_q = OrderedDict(sorted(last_q.items()), key=lambda i: keyorder.index(i[0]))
    if 'key' in sorted_q:
        del sorted_q['key']

    # plot_idea_cluster('Cluster Number')
    # plot_idea_cluster('Cluster Size')
    # plot_idea_cluster('(Cluster Size, Cluster Number)')
    plot_ideal_q_value(sorted_q, last_epi)
    # Initial Ideal Q value
    # for k, v in sorted_q.items():
    #     # (cs, cn)
    #     if k[0] < 10:
    #         sorted_q[k] = [0, 1, 0, 0, 0]
    #     elif k[0] == 10 and k[1] < 10:
    #         sorted_q[k] = [0, 0, 0, 1, 0]
    #     else:
    #         sorted_q[k] = [1, 0, 0, 0, 0]
    # gym_plt.v1_plot_action_value(sorted_q, title="Experiment 1: Action Values Overview - {} episodes".format(last_epi+1))

    # preference_action = []
    # for k, v in sorted_q.items():
    #     preference_action.append(v.index(max(v)))
    # X = [x+1 for x in range(len(preference_action))]
    #
    # plt.xlabel("Action")
    # plt.ylabel("Preference Action")
    # plt.ylim([0, 100])
    # # plt.axhspan(13, 33, alpha=0.5)
    # plt.plot(X, preference_action)
    # plt.show()


if __name__ == '__main__':
    sys.exit(main())
