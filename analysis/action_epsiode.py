import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import os
import json
from collections import namedtuple

csv.field_size_limit(sys.maxsize)


def plot_episode_analysis(data, reward_data, epi):
    X = list(range(1, len(data)+1, 1))
    plt.clf()
    plt.xlabel("Cycle")
    plt.ylabel("Action Taken")
    plt.title("Action Taken in {} Episode".format(epi))
    plt.ylim([0, 11])
    plt.axhspan(9, 11, alpha=0.5)
    plt.grid()
    plt.plot(X, data, label="Action Taken")
    plt.plot(X, reward_data, label="Rewards")
    plt.legend(loc="lower left")
    plt.show()


def plot_action_distribution(data):
    fig = plt.figure(1, figsize=(40, 15))
    ax = fig.add_subplot(111)

    plt.xlabel("Episode")
    plt.ylabel("Action Distribution")
    plt.title('Action distribution over episode')
    ax.boxplot(data, showfliers=False)
    plt.show()


def main():
    # {epi: {state: [], action, reward}}

    x = {}
    boxplot_data = {}
    with open('../agents/records/exp-3-epi-100-train-0-maintain-all-terminal-100-2.csv') as f:
        # with open('./data/exp4/exp-4-training-epi-200-vm-100.csv') as f:
        reader = csv.DictReader(f)
        for line in reader:
            epi = int(line['episode']) + 1
            if epi not in x:
                x[epi] = {"state": [], "action": [], "reward": []}
                boxplot_data[epi] = []
            x[epi]["state"].append(int(line["state"]))
            x[epi]["action"].append(int(line["action"]) + 1)
            x[epi]["reward"].append(float(line["reward"]))
            boxplot_data[epi].append(int(line["action"]) + 1)

    # plot_action_distribution(boxplot_data.values())
    last_epi = 61  # len(x)
    plot_episode_analysis(x[last_epi]["action"], x[last_epi]["reward"], last_epi-1)
    print(sum(x[last_epi]["reward"]))
    # reward counter
    # for k, v in x.items():
    #     print("reward counter {}".format(len(list(filter(lambda x: x > 0, v["reward"])))))
    #     print("correct action counter {}".format(len(list(filter(lambda x: x > 7, v["action"])))))


if __name__ == '__main__':
    sys.exit(main())
