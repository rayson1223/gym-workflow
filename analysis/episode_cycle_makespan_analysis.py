import sys
import csv
import os
import json
import matplotlib.pyplot as plt


def main():
    x = {}
    all = []
    with open('../agents/records/v10-training-epi-100-vm-10-train-10.csv_execution_records.csv') as f:
        reader = csv.DictReader(f)
        for line in reader:
            epi = int(line['episode']) + 1
            if epi not in x:
                x[epi] = []
            makespan = json.loads(line['records'])['makespan']
            for r in makespan:
                x[epi].append(float(r))
                all.append(float(r))


    # plt.plot(x[10], ':', label='Episode 10')
    # # plt.plot(x[20], '-.', label='Episode 20')
    # plt.plot(x[30], '--', label='Episode 30')
    # # plt.plot(x[40], '-.', label='Episode 40')
    # plt.plot(x[50], '-.^', label='Episode 50')
    # plt.plot(x[100], '-', label='Episode 90')
    # plt.legend()
    # # Add title and x, y labels
    # plt.title("Makespan analysis within episode")
    # plt.xlabel("Makespan(s)")
    # plt.ylabel("Cycle(s)")
    # plt.savefig("./plots/makespan-v10-epi-100-train-0.png")
    # plt.show()
    plt.figure(figsize=(50,5))
    plt.plot(all)
    plt.show()


if __name__ == '__main__':
    sys.exit(main())
