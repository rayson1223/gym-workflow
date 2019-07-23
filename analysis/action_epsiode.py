import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import os
import json
from collections import namedtuple

csv.field_size_limit(sys.maxsize)


def main():
    # {epi: {state: [], action, reward}}

    x = {}
    boxplot_data = {}
    # with open('../agents/records/v10-training-epi-50-vm-10.csv') as f:
    with open('./data/exp4/exp-4-training-epi-200-vm-100.csv') as f:
        reader = csv.DictReader(f)
        for line in reader:
            epi = int(line['episode']) + 1
            if epi not in x:
                x[epi] = {"state": [], "action": [], "reward": []}
                boxplot_data[epi] = []
            x[epi]["state"].append(int(line["state"]))
            x[epi]["action"].append(int(line["action"]))
            x[epi]["reward"].append(float(line["reward"]))
            boxplot_data[epi].append(int(line["action"]))

    fig = plt.figure(1, figsize=(40, 15))
    ax = fig.add_subplot(111)

    plt.xlabel("Episode")
    plt.ylabel("Action Distribution")
    plt.title('Action distribution over episode')

    ax.boxplot(boxplot_data.values(), showfliers=False)

    plt.show()

    # reward counter
    for k, v in x.items():
        print("reward counter {}".format(len(list(filter(lambda x: x > 0, v["reward"])))))
        print("correct action counter {}".format(len(list(filter(lambda x: x > 5, v["action"])))))


if __name__ == '__main__':
    sys.exit(main())
