import json
import sys
import matplotlib.pyplot as plt
import numpy as np


def main():
    x = {}
    with open('./records/workflowsim_analysis_record_cn_100_collect_30-final.csv') as f:
        x = json.load(f)

    file_name = "workflowsim-cn_100_collect_100_delay_100"
    # Process plotting on each key
    for t in x["1"].keys():
        temp = []
        # get all the values for related key in each episode
        # format: [[1...], [2...]]
        for epi in x.keys():
            temp.append(x[epi][t])

        # get the median in each episode
        median_temp = []
        for et in temp:
            median_temp.append(np.mean(et))
        # Remove 1 because no 1 too big
        # median_temp.remove(median_temp[0])
        median_temp[0] = median_temp[1] + median_temp[2]
        # Clear cache
        plt.clf()
        if len(list(filter(lambda x: len(x) > 0, temp))) > 0:
            # plt.figure(figsize=(40, 15))
            # plt.xlabel("Episode")
            # plt.ylabel("{}(s)".format(t))
            # plt.title("{} over episode".format(t))
            #
            # plt.boxplot(temp, showfliers=False)
            # plt.savefig("./plots/{}-{}.png".format(file_name, t))
            # plt.show()
            if t == "makespan":
                plot_median(median_temp, "Makespan (mean) over episode".format(t), file_name, label=t)


def plot_median(data, title="", file_name="median", label=""):
    # find the highlight region
    vl = list(filter(lambda x: x < np.percentile(data, 10), data))
    vli = [data.index(x) for x in vl]
    print(data)
    print("{}: min-{}, max-{}".format(label, min(vli), max(vli)))
    plt.clf()
    # plt.xticks(np.arange(1, 101, step=1))
    plt.xlabel("Cluster Number")
    plt.ylabel("Mean {}".format(label))
    # plt.title(title)
    plt.axvspan(min(vli), max(vli), alpha=0.5)
    plt.grid()
    plt.plot(np.arange(1, 101), data)
    plt.savefig("./plots/{}-{}-median.png".format(file_name, label))
    plt.show()


if __name__ == '__main__':
    sys.exit(main())
