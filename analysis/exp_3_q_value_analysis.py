import PyGnuplot as gp
import numpy as np
import csv
from collections import OrderedDict
import json
import matplotlib.pyplot as plt
import sys
import ast
import agents.utils.plotting as gym_plt

csv.field_size_limit(sys.maxsize)


def identified_ideal_result():
    makespan_record = {}
    with open('./data/all_cs_results.csv') as f:
        reader = csv.DictReader(f)
        for line in reader:
            degree = line['degrees']
            cs = line['cluster_size']
            if degree not in makespan_record:
                makespan_record[degree] = {}
            if cs not in makespan_record[degree]:
                makespan_record[degree][cs] = []
            makespan_record[degree][cs].append(line['exec_time'])

    def visualize(data):
        all_data = []
        all_median = []
        for cs, v in data.items():
            for i, d in enumerate(v):
                v[i] = int(d)
            all_data += v
            all_median.append(np.median(v))
        ti = []
        for i, v in enumerate(all_median):
            if v < np.percentile(all_data, 20):
                ti.append(i + 1)
        x = []
        for v in data.keys():
            x.append(int(v))
        plt.xlabel("Cluster Size")
        plt.ylabel("Makespan")
        # plt.ylim([0, 100])
        plt.axvspan(min(ti), max(ti), alpha=0.5)
        plt.boxplot(data.values())
        # plt.boxplot(data.values())
        plt.show()

    return visualize(makespan_record['0.5'])


def ideal_q_value(Q={}):
    X = range(1, 11, 1)
    Y = np.zeros(10)
    Y.fill(10)

    def line_version(X, Y):
        plt.xlabel("State (cluster size)")
        plt.ylabel("Q value (preference action)")
        plt.ylim([0, 11])
        plt.axhspan(8, 11, alpha=0.5)
        plt.plot(X, Y)
        plt.show()

    def mesh_version(Q, value=[0]):
        for k, v in Q.items():
            for i, m in enumerate(v):
                if i in value:
                    v[i] = 1
                else:
                    v[i] = 0
        gym_plt.plot_exp_2_action_value(Q)

    line_version(X, Y)
    mesh_version(Q, [8,9])


def epi_analysis(data, episode=100):
    gym_plt.plot_exp_2_action_value(data, title="Experiment 3: Q-values overview - {} episodes".format(episode))
    preference_action = []
    for k, v in data.items():
        preference_action.append(v.index(max(v)) + 1)
    X = [x + 1 for x in range(len(preference_action))]

    plt.xlabel("State (cluster size)")
    plt.ylabel("Q value (preference action)")
    # plt.ylim([0, 10])
    plt.axhspan(8, 11, alpha=0.5)
    plt.ylim([0, 11])
    plt.plot(X, preference_action)
    plt.title("Q-value (preference action) over state")
    plt.show()



def main():
    # identified_ideal_result()
    q_record = {}
    with open('../agents/records/exp-3-epi-300-train-0-maintain-all-terminal-200-M11-B10.csv_episode_q_value.csv') as f:
        reader = csv.DictReader(f)
        for line in reader:
            fs = line['Q Value'].replace("array(", "").replace("])", "]").replace("\n", "")
            ans = ast.literal_eval(fs)
            # print(ans)
            q_record[int(line['episode'])] = ans
            # IT FUCKING WORKS
            # plt.plot_exp_3_action_value(ans)

    last_epi = len(q_record) - 1
    last_q = q_record[last_epi]

    sorted_q = OrderedDict(sorted(last_q.items()), key=lambda i: keyorder.index(i[0]))
    if 'key' in sorted_q:
        del sorted_q['key']

    # identified_ideal_result()
    # ideal_q_value(last_q)
    epi_analysis(last_q)


if __name__ == '__main__':
    sys.exit(main())
