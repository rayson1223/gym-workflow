import csv
import json
import sys

import matplotlib.pyplot as plt

csv.field_size_limit(sys.maxsize)


def main():
    # Mainly used to analyze what's the parameters we should set in order to have a predictable results for cluster size
    # Data required:
    #   - Episode
    #   - Action
    #   - Makespan
    #   - Overhead
    # Goal: determine a good hyper-parameter that cluster size will not have big effect over the overhead and makespan
    #

    # Data reading phase
    makespan_records = {}
    overhead_records = {}
    action_records = {}
    overhead_final = {}
    makespan_final = {}

    # Episode Overhead & Makespan Data loading
    with open('./data/exp3/v8-training-epi-300-vm-100.csv_execution_records.csv') as f:
        reader = csv.DictReader(f)
        for line in reader:
            epi = int(line['episode']) + 1
            if epi not in makespan_records:
                makespan_records[epi] = []
                overhead_records[epi] = []
            records = json.loads(line['records'])
            makespan = records['makespan']
            overhead = records['overhead']
            for r in makespan:
                makespan_records[epi].append(float(r))
            for o in overhead:
                overhead_records[epi].append(float(o))

    # Episode action loading
    with open('./data/exp3/v8-training-epi-300-vm-100.csv') as f:
        reader = csv.DictReader(f)
        for line in reader:
            epi = int(line['episode']) + 1
            if epi not in action_records:
                action_records[epi] = {"state": [], "action": [], "reward": []}
            action_records[epi]["state"].append(int(line["state"]))
            action_records[epi]["action"].append(int(line["action"]))
            action_records[epi]["reward"].append(float(line["reward"]))

    # Map the action -> makespan & overhead
    # example:
    #   {action: [records]}
    #
    # Loop logic:
    #   - Loop through episode -> get action map with the cycle index -> dump in record according to action
    #

    for epi in makespan_records.keys():
        for i, action in enumerate(action_records[epi]["action"]):
            action += 1
            if action not in overhead_final:
                overhead_final[action] = []
                makespan_final[action] = []
            makespan_final[action].append(makespan_records[epi][i])
            overhead_final[action].append(overhead_records[epi][i])


    # Process ploting
    fig = plt.figure(1, figsize=(40, 15))
    ax = fig.add_subplot(111)

    plt.xlabel("Action")
    plt.ylabel("Makespan(s)")
    plt.title('Makespan distribution over Actions')

    ax.boxplot(makespan_final.values(), showfliers=True)

    plt.show()

    fig2 = plt.figure(1, figsize=(40, 15))
    ax = fig2.add_subplot(111)

    plt.xlabel("Action")
    plt.ylabel("Overhead(s)")
    plt.title('Overhead distribution over Actions')

    ax.boxplot(overhead_final.values(), showfliers=True)

    plt.show()


if __name__ == '__main__':
    sys.exit(main())
