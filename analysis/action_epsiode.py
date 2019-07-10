import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import os
import agents.utils.plotting as draw
import json
from collections import namedtuple

csv.field_size_limit(sys.maxsize)


def main():
    # {epi: {state: [], action, reward}}

    x = {}
    with open('../agents/records/v10-training-epi-50-vm-10.csv') as f:
        reader = csv.DictReader(f)
        for line in reader:
            epi = int(line['episode']) + 1
            if epi not in x:
                x[epi] = {"state": [], "action": [], "reward": []}
            x[epi]["state"].append(int(line["state"]))
            x[epi]["action"].append(int(line["action"]))
            x[epi]["reward"].append(float(line["reward"]))

    # reward counter
    for k, v in x.items():
        print("reward counter {}".format(len(list(filter(lambda x: x > 0, v["reward"])))))
        print("correct action counter {}".format(len(list(filter(lambda x: x > 5, v["action"])))))


if __name__ == '__main__':
    sys.exit(main())
