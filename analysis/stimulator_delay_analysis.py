import json
import sys
import matplotlib.pyplot as plt


def main():
    x = {}
    with open('./records/workflowsim_analysis_record_cs_100_collect_100_delay_10.csv') as f:
        x = json.load(f)

    file_name = "workflowsim-cs_100_collect_100_delay_10"
    # Process plotting on each key
    for t in x["1"].keys():
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
