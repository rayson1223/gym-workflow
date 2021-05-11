import numpy as np
import pandas as pd
import csv
from collections import OrderedDict
import json
import matplotlib.pyplot as plt
import sys
import ast
import agents.utils.plotting as gym_plt

csv.field_size_limit(sys.maxsize)


def plot_reward():
    preference_action = [9, 10, 9, 10, 10, 9, 10, 10, 9, 9]
    X = [x + 1 for x in range(len(preference_action))]
    fig, ax = plt.subplots(figsize=(10, 7))

    plt.xlabel("State (cluster size)", fontsize=22)
    plt.ylabel("Q value (preference action)", fontsize=22)
    # plt.ylim([0, 10])
    ax.axhspan(8, 11, alpha=0.5)
    plt.xlim([0, 10.5])
    plt.ylim([0, 10.5])
    ax.set_xticks(np.arange(0, 11, 1))
    ax.set_yticks(np.arange(0, 11, 1))
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    ax.plot(X, preference_action)
    plt.title("Q-value (preference action) over state")
    plt.show()


def main():
    # rewards = {}
    #
    # with open('../agents/records/publication/p3-exp1-training-epi-100-vm-100-v12-B10-final.csv') as f:
    #     reader = csv.DictReader(f)
    #     for line in reader:
    #         if not (int(line['episode']) + 1) in rewards:
    #             rewards[int(line['episode']) + 1] = 0
    #         rewards[int(line['episode']) + 1] += int(line['reward'])
    # print(line['episode'])
    # print()
    # break
    # print(rewards.values())
    plot_reward()


if __name__ == '__main__':
    sys.exit(main())
