from gym import make
import gym_workflow.envs
import sys
import csv
import os
import json
import matplotlib.pyplot as plt


def main():
    env = make('Montage-v12')

    # Data Format
    # {
    #   action: {
    #       makespan: [],
    #       overhead: [],
    #       delay etc
    #   }
    # }
    records = {}
    cs_range = 100
    collector_range = 30
    for i in range(cs_range):
        records[i] = {
            'makespan': [],
            'queueDelay': [],
            'execTime': [],
            'postscriptDelay': [],
            'clusterDelay': [],
            'WENDelay': []
        }
        for j in range(collector_range):
            print("\r Cluster Number {}/{}, Sampling {}/{}".format(i + 1, cs_range, j+1, collector_range), end="")
            sys.stdout.flush()
            state = env.reset()
            next_state, reward, done, exec_record = env.step(i, training=False)
            records[i]['makespan'].append(float(exec_record['makespan']))
            records[i]['queueDelay'].append(float(exec_record['queue']))
            records[i]['execTime'].append(float(exec_record['exec']))
            records[i]['postscriptDelay'].append(float(exec_record['postscript']))
            records[i]['clusterDelay'].append(float(exec_record['cluster']))
            records[i]['WENDelay'].append(float(exec_record['wen']))

    file_name = "workflowsim_analysis_record_cn_{}_collect_{}-publication.csv".format(cs_range, collector_range)
    # if not os.path.exists(os.getcwd() + '/records/' + file_name):
    #     with open(os.getcwd() + '/records/' + file_name, 'w', newline='', encoding='utf-8') as r:
    #         writer = csv.DictWriter(r, ['records'])
    #         writer.writeheader()

    with open(os.getcwd() + '/records/' + file_name, 'w') as r:
        json.dump(records, r)

    # # reformat the data for re-ploting graph
    # makespanV = []
    # queueV = []
    # execV = []
    # postV = []
    # clusterV = []
    # WENV = []
    # for i in range(records):
    #     makespanV.append(records[i]['makespan'])
    #     queueV.append(records[i]['queueDelay'])
    #     execV.append(records[i]['execTime'])
    #     postV.append(records[i]['postscriptDelay'])
    #     clusterV.append(records[i]['clusterDelay'])
    #     WENV.append(records[i]['WENDelay'])
    # Process ploting
    # fig = plt.figure(1, figsize=(40, 15))
    # ax = fig.add_subplot(111)
    #
    # plt.xlabel('Action')
    # plt.ylabel('Makespan(s)')
    # plt.title('Makespan distribution over Actions')
    #
    # ax.boxplot(makespanV, showfliers=False)
    #
    # plt.show()
    #
    # fig2 = plt.figure(1, figsize=(40, 15))
    # ax = fig2.add_subplot(111)
    #
    # plt.xlabel('Action')
    # plt.ylabel('Queue Delay(s)')
    # plt.title('Overhead distribution over Actions')
    #
    # ax.boxplot(queueV, showfliers=True)
    #
    # plt.show()
    #
    # fig2 = plt.figure(1, figsize=(40, 15))
    # ax = fig2.add_subplot(111)
    #
    # plt.xlabel('Action')
    # plt.ylabel('Overhead(s)')
    # plt.title('Overhead distribution over Actions')
    #
    # ax.boxplot(overhead_final.values(), showfliers=True)
    #
    # plt.show()
    #
    # fig2 = plt.figure(1, figsize=(40, 15))
    # ax = fig2.add_subplot(111)
    #
    # plt.xlabel('Action')
    # plt.ylabel('Overhead(s)')
    # plt.title('Overhead distribution over Actions')
    #
    # ax.boxplot(overhead_final.values(), showfliers=True)
    #
    # plt.show()
    #
    # fig2 = plt.figure(1, figsize=(40, 15))
    # ax = fig2.add_subplot(111)
    #
    # plt.xlabel('Action')
    # plt.ylabel('Overhead(s)')
    # plt.title('Overhead distribution over Actions')
    #
    # ax.boxplot(overhead_final.values(), showfliers=True)
    #
    # plt.show()
    #
    # fig2 = plt.figure(1, figsize=(40, 15))
    # ax = fig2.add_subplot(111)
    #
    # plt.xlabel('Action')
    # plt.ylabel('Overhead(s)')
    # plt.title('Overhead distribution over Actions')
    #
    # ax.boxplot(overhead_final.values(), showfliers=True)
    #
    # plt.show()


if __name__ == '__main__':
    sys.exit(main())
