import csv
import json
import sys

import matplotlib.pyplot as plt

csv.field_size_limit(sys.maxsize)


def main():
    x = {}
    with open('../agents/records/p1_3-training-0-epi-100-vm-10.csv_execution_records.csv') as f:
    # with open('./data/exp4/exp-4-training-epi-200-vm-100.csv_execution_records.csv') as f:
        reader = csv.DictReader(f)
        for line in reader:
            epi = int(line['episode']) + 1
            if epi not in x:
                x[epi] = {
                    "exec": [],
                    "queueDelay": [],
                    "overhead": [],
                    "makespan": [],
                    "postscriptDelay": [],
                    "clusterDelay": [],
                    "WENDelay": [],
                    "benchmark": []
                }
            records = json.loads(line['records'])
            x[epi]["exec"] = records['exec']
            x[epi]["queueDelay"] = records['queueDelay'] if 'queueDelay' in records else None
            x[epi]["overhead"] = records['overhead']
            x[epi]["makespan"] = records['makespan']
            x[epi]["postscriptDelay"] = records['postscriptDelay'] if 'postscriptDelay' in records else None
            x[epi]["clusterDelay"] = records['clusterDelay'] if 'clusterDelay' in records else None
            x[epi]["WENDelay"] = records['WENDelay'] if 'WENDelay' in records else Nonef
            x[epi]["benchmark"] = records['benchmark']

            # for r in makespan:
            #     x[epi].append(float(r))

    # Data conversion
    for epi in x:
        for t in x[epi].keys():
            for k, v in enumerate(x[epi][t]):
                x[epi][t][k] = float(v)

    file_name = "p1_3_analysis"
    # Process plotting on each key
    for t in x[1].keys():
        temp = []
        for epi in x.keys():
            temp.append(x[epi][t])

        # Clear cache
        plt.clf()
        if len(list(filter(lambda x: len(x) > 0, temp))) > 0:
            plt.figure(figsize=(40, 15))
            plt.xlabel("Episode")
            plt.ylabel("{}(s)".format(t))
            plt.title("{} over episode".format(t))

            plt.boxplot(temp, showfliers=False)
            plt.savefig("./plots/{}-{}.png".format(file_name, t))
            plt.show()


if __name__ == '__main__':
    sys.exit(main())
