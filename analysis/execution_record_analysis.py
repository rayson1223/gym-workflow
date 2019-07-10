import csv
import json
import sys

import matplotlib.pyplot as plt

csv.field_size_limit(sys.maxsize)


def main():
    x = {}
    with open('../agents/records/v8-training-epi-100-vm-100.csv_execution_records.csv') as f:
        reader = csv.DictReader(f)
        for line in reader:
            epi = int(line['episode']) + 1
            if epi not in x:
                x[epi] = []
            makespan = json.loads(line['records'])['makespan']
            for r in makespan:
                x[epi].append(float(r))

    # Process ploting
    # fig1, ax1 = plt.subplots()
    fig = plt.figure(1, figsize=(40, 15))
    ax = fig.add_subplot(111)

    plt.xlabel("Episode")
    plt.ylabel("Makespan(s)")
    plt.title('Makespan distribution over episode')

    ax.boxplot(x.values(), showfliers=False)

    plt.show()


if __name__ == '__main__':
    sys.exit(main())
