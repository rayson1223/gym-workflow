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


def plot_reward(data):
    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10, 5))
    rewards_smoothed = pd.Series(data).rolling(10, min_periods=10).mean()
    plt.plot(rewards_smoothed)
    # plt.plot(data.values())
    # plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim([0, 100])
    # plt.axhspan(20, 25, alpha=0.5)
    # plt.hlines(20, xmin=0, xmax=100, linestyles='dashed')
    plt.xlabel("Episode", fontsize=16)
    plt.ylabel("Episode Reward", fontsize=16)
    plt.title("Episode Reward over Time (Smoothed over window size 10)")
    # fig2.savefig("O".format(10))
    fig2.show()


def main():
    rewards = {}

    # with open('../agents/records/p2-exp1-epi-100-train-0-maintain-all-terminal-200-M10-B20.csv') as f:
    with open('../agents/records/p3-exp1-training-epi-100-vm-100-v12-B10-final.csv') as f:
        # with open('./data/exp4/exp-4-training-epi-200-vm-100.csv_episode_q_value.csv') as f:
        reader = csv.DictReader(f)
        for line in reader:
            if not (int(line['episode']) + 1) in rewards:
                rewards[int(line['episode']) + 1] = 0
            rewards[int(line['episode']) + 1] += int(line['reward'])
            # print(line['episode'])
            # print()
            # break
    # print(rewards.values())
    plot_reward(rewards)


if __name__ == '__main__':
    sys.exit(main())
