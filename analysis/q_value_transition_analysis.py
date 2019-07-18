from matplotlib import pyplot as plt
import csv
import json
import sys


def main():
    with open('../agents/records/exp-3-epi-50-train-0-maintain-all_episode_q_value.csv') as f:
        reader = csv.DictReader(f)
        for line in reader:
            fs = line['Q Value'].replace("array(", "").replace("])", "]")
            print(fs)
            print(json.loads(fs))


if __name__ == '__main__':
    sys.exit(main())
