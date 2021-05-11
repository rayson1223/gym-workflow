import numpy as np
import csv
from collections import OrderedDict
import json
import matplotlib.pyplot as plt
import sys
import ast
import agents.utils.plotting as gym_plt

csv.field_size_limit(sys.maxsize)


def ideal_q_value(Q={}):
    X = range(1, 101, 1)
    Y = np.zeros(100)
    Y.fill(6)

    # vl = list(filter(lambda x: x < np.percentile(data, 10), data))
    def line_version(X, Y):
        plt.xlabel("State (cluster number)")
        plt.ylabel("Q-value (preference action)")
        # plt.title("Q-value (preference action) over state")
        plt.ylim([0, 100])
        plt.axhspan(2, 11, alpha=0.5)
        plt.plot(X, Y)
        plt.show()

    def mesh_version(Q, value=[0]):
        for k, v in Q.items():
            for i, m in enumerate(v):
                if i in list(range(value[0], value[1] + 1, 1)):
                    v[i] = 1
                else:
                    v[i] = 0
        gym_plt.plot_exp_3_action_value(Q, title="", ticks=False)

    line_version(X, Y)
    mesh_version(Q, [2, 11])


def epi_analysis(data):
    # gym_plt.plot_exp_3_action_value(data, title="Q-value overview - 100 episodes", ticks=False)
    preference_action = []
    for k, v in data.items():
        preference_action.append(v.index(max(v)))
    X = [x + 1 for x in range(len(preference_action))]

    # Count the total no of action that in range
    count = 0
    for v in preference_action:
        # print(v)
        if 2 <= v <= 11:
            count += 1
    print("Count is: ", count)
    plt.xlabel("State (cluster number)")
    plt.ylabel("Q value (preference action)")
    plt.ylim([0, 100])
    # WHY IS 2 to 6
    plt.axhspan(2, 11, alpha=0.5)
    plt.title("Q-value (preference action) over state")
    plt.plot(X, preference_action)
    plt.show()


def main():
    q_record = {}

    with open('../agents/records/p3-exp1-training-epi-100-vm-100-v12-B10-final.csv_episode_q_value.csv') as f:
        # with open('./data/exp4/exp-4-training-epi-200-vm-100.csv_episode_q_value.csv') as f:
        reader = csv.DictReader(f)
        for line in reader:
            fs = line['Q Value'].replace("array(", "").replace("])", "]").replace("\n", "")
            # for num in range(10):
            #     fs = fs.replace("{}:".format(num+1), "\"{}\":".format(num+1))
            ans = ast.literal_eval(fs)
            q_record[int(line['episode'])] = ans
            # IT FUCKING WORKS
            # plt.plot_exp_3_action_value(ans)

    last_epi = len(q_record) - 1
    last_q = q_record[last_epi]

    sorted_q = OrderedDict(sorted(last_q.items()), key=lambda i: keyorder.index(i[0]))
    if 'key' in sorted_q:
        del sorted_q['key']

    # ideal_q_value(sorted_q)
    # epi_analysis(sorted_q)

    # gym_plt.plot_exp_3_action_value(sorted_q, title="Q-value overview - 100 episodes", ticks=False)


if __name__ == '__main__':
    sys.exit(main())
