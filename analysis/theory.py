import sys
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import random
from functools import reduce
import json
import csv
from gym_workflow.libs.recording import write_record
from gym_workflow.libs.montage.db.model.pegasus_wf import PegasusWf


def gen_cn_experiment(vmSize=20, cluster_method="NONE", cluster_size=1):
    cmd = "java -jar ../gym_workflow/libs/workflowsim/WorkflowSim-cn.jar {} {} {} {}".format(
        vmSize, cluster_method, cluster_size, "../gym_workflow/libs/workflowsim/dax/Montage_1000.xml")
    output = subprocess.getoutput(cmd).strip().split('\n')
    return output[len(output) - 1].split()


def gen_cs_experiment(vmSize=20, cluster_method="NONE", cluster_size=1):
    cmd = "java -jar ../gym_workflow/libs/workflowsim/WorkflowSim-cs.jar {} {} {} {}".format(
        vmSize, cluster_method, cluster_size, "../gym_workflow/libs/workflowsim/dax/Montage_1000.xml")
    output = subprocess.getoutput(cmd).strip().split('\n')
    return output[len(output) - 1].split()


def plot_all_makespan_overhead():
    makespanN = {}
    makespanH = {}
    makespanV = {}
    makespanB = {}

    queueN = {}
    queueH = {}
    queueV = {}
    queueB = {}

    execN = {}
    execH = {}
    execV = {}
    execB = {}

    postN = {}
    postH = {}
    postV = {}
    postB = {}

    costN = {}
    costH = {}
    costV = {}
    costB = {}

    overheadN = {}
    overheadH = {}
    overheadV = {}
    overheadB = {}

    vmSize = random.randint(1, 30)

    for i in range(3, 30):
        # Get results
        msN, qN, eN, pN, cN, vmN = gen_cn_experiment(vmSize, cluster_method="NONE")
        msH, qH, eH, pH, cH, vmH = gen_cn_experiment(vmSize, cluster_method="HORIZONTAL", cluster_size=i)
        # msV, qV, eV, pV, cV, vmV = gen_cn_experiment(cluster_method="VERTICAL", cluster_size=i)
        # msB, qB, eB, pB, cB, vmB = gen_cn_experiment(cluster_method="BALANCED", cluster_size=i)

        # Assign to appropriate var
        makespanN[i] = float(msN)
        makespanH[i] = float(msH)
        # makespanV[i] = float(msV)
        # makespanB[i] = float(msB)

        queueN[i] = float(qN)
        queueH[i] = float(qH)
        # queueV[i] = float(qV)
        # queueB[i] = float(qB)

        execN[i] = float(eN)
        execH[i] = float(eH)
        # execV[i] = float(eV)
        # execB[i] = float(eB)

        postN[i] = float(pN)
        postH[i] = float(pH)
        # postV[i] = float(pV)
        # postB[i] = float(pB)

        costN[i] = float(cN)
        costH[i] = float(cH)
        # costV[i] = float(cV)
        # costB[i] = float(cB)

        overheadN[i] = queueN[i] + postN[i]
        overheadH[i] = queueH[i] + postH[i]
        # overheadV[i] = queueV[i] + queueV[i]
        # overheadB[i] = queueB[i] + queueB[i]
    fig, ax = plt.subplots()

    tempO = list(overheadH.values())
    tempM = list(makespanH.values())

    ax.plot(makespanN.keys(), makespanN.values(), 'k', label="nc makespan")
    ax.plot(makespanH.keys(), makespanH.values(), 'b', label="hc makespan")
    # ax.plot(makespanV.keys(), makespanV.values(), 'r', label="vc makespan")
    # ax.plot(makespanB.keys(), makespanB.values(), 'g', label="bc makespan")

    # ax.plot(queneN.keys(), queueN.values(), 'k:', label="nc queue")
    # ax.plot(queueH.keys(), queueH.values(), 'b:', label="hc queue")
    # ax.plot(queueV.keys(), queueV.values(), 'r:', label="vc queue")
    # ax.plot(queueB.keys(), queueB.values(), 'g:', label="bc queue")

    ax.plot(execN.keys(), execN.values(), 'k-.', label="nc exec")
    ax.plot(execH.keys(), execH.values(), 'b-.', label="hc exec")
    # ax.plot(execV.keys(), execV.values(), 'r-.', label="vc exec")
    # ax.plot(execB.keys(), execB.values(), 'g-.', label="bc exec")

    # ax.plot(postN.keys(), postN.values(), 'k--', label="nc postscripted")
    # ax.plot(postH.keys(), postH.values(), 'b--', label="hc postscripted")
    # ax.plot(postV.keys(), postV.values(), 'r--', label="vc postscripted")
    # ax.plot(postB.keys(), postB.values(), 'g--', label="bc postscripted")

    ax.plot(overheadN.keys(), overheadN.values(), 'k--', label="nc overhead")
    ax.plot(overheadH.keys(), tempO, 'b--', label="hc overhead")
    # ax.plot(overheadV.keys(), overheadV.values(), 'r--', label="vc overhead")
    # ax.plot(overheadB.keys(), overheadB.values(), 'g--', label="bc overhead")

    ax.plot(costN.keys(), costN.values(), 'k:', label="nc cost")
    ax.plot(costH.keys(), costH.values(), 'b:', label="hc cost")

    # plt.plot(costN.keys(), costN.values(), label="non-clustering cost")
    # plt.plot(costH.keys(), costH.values(), label="horizontal-clustering cost")
    ax.legend(loc="upper left")
    plt.grid(True)
    plt.xlabel("cluster size")
    plt.ylabel("time (s)")
    plt.show()
    print("VM Size is: {}".format(vmSize))
    print("Analysis:")
    print("Horizontal Overhead Variance: {}".format(np.var(tempO)))
    print("Horizontal makespan Variance: {}".format(np.var(tempM)))


def overhead_analysis(cluster_method="HORIZONTAL", cluster_size=20):
    data = {}
    # Data Collection
    for cs in range(cluster_size):
        data[cs + 1] = []
        print("Processing cluster size of: {}".format(cs))
        for j in range(30):
            result = gen_cs_experiment(cluster_method=cluster_method, cluster_size=cs + 1)
            data[cs + 1].append((float(result[1]) + float(result[3])))

    # Process the percentile analysis
    k, v = zip(*data.items())
    allV = reduce(lambda x, y: x + y, v)
    p20 = np.percentile(allV, 20)
    p30 = np.percentile(allV, 30)
    p40 = np.percentile(allV, 40)
    p50 = np.percentile(allV, 50)
    p20x = [p20] * len(k)
    p30x = [p30] * len(k)
    p40x = [p40] * len(k)
    p50x = [p50] * len(k)

    write_record([allV], header=['overhead', 'p-20', 'p-30', 'p-40', ''])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.boxplot(data.values(), labels=data.keys())
    ax.plot(data.keys(), p20x, label="p-20")
    ax.plot(data.keys(), p30x, label="p-30")
    ax.plot(data.keys(), p40x, label="p-40")
    ax.plot(data.keys(), p50x, label="p-50")
    ax.legend(loc="upper right")
    ax.set_title("Cluster Size vs Overhead(s)")
    ax.set_xlabel("Cluster Size")
    ax.set_ylabel("Overhead(s)")
    plt.show()


def plot_cs_analysis():
    data = {}
    with open('./data/fundamental/final_cs_result_30.csv') as f:
        reader = csv.DictReader(f)
        for line in reader:
            cs = int(line['cluster_size'])
            if cs not in data:
                data[cs] = []
            data[cs].append(float(line['wall_time']))
    get_mean_results(data)
    plt.clf()
    fig = plt.figure(1, figsize=(40, 20))
    ax = fig.add_subplot(111)

    plt.xlabel("Cluster Size")
    plt.ylabel("Makespan mean")
    # plt.title('Makespan over cluster size')
    ax.boxplot(data.values(), showfliers=False)
    plt.grid()
    plt.savefig('./plots/publication-cs-analysis.png')
    plt.show()


def plot_cn_analysis():
    data = {}
    with open('./data/fundamental/montage_num_1to30_sample.csv') as f:
        reader = csv.DictReader(f)
        for line in reader:
            cs = int(line['cluster_num'])
            if cs not in data:
                data[cs] = []
            data[cs].append(float(line['wall_time']))
    mean_results = get_mean_results(data)
    plt.clf()
    fig = plt.figure(1, figsize=(30, 15))
    ax = fig.add_subplot(111)
    plt.xlabel("Cluster Number")
    plt.ylabel("Mean Makespan")
    # plt.title('Makespan(s) over cluster number')
    plt.axvspan(8, 11, alpha=0.5)
    plt.grid()
    # ax.boxplot(data.values(), showfliers=False)
    ax.plot(mean_results.keys(), mean_results.values())
    # plt.savefig('./plots/final-cn-analysis.png')
    plt.show()


def plot_cs_cn_analysis():
    data = {}
    label = []
    with open('./data/fundamental/cs_cn_analysis.csv') as f:
        reader = csv.DictReader(f)
        for line in reader:
            cs = int(line['cluster_size'])
            cn = int(line['cluster_num'])
            if cs == 3:
                continue
            l = "({}, {})".format(cs, cn)
            if l not in data:
                data[l] = []
            data[l].append(float(line['wall_time']))
            # if cs not in data:
            #     data[cs] = {}
            # if cn not in data[cs]:
            #     data[cs][cn] = []
            # data[cs][cn].append(float(line['exec_time']))

    plt.clf()
    plt.figure(figsize=(15, 10))
    plt.boxplot(data.values(), showfliers=False)
    plt.xlabel("(cluster size, cluster Number)")
    plt.ylabel("Makespan(s)")
    plt.title('Makespan(s) over (cluster size, cluster Number)')
    print(data.keys())
    plt.xticks(range(len(data.keys())), labels=data.keys(), rotation='vertical')
    plt.savefig('./plots/publication-cs-cn-analysis.png')
    plt.show()


def get_wf_results(dir):
    wf = PegasusWf()
    wf.initialize_by_work_dir(dir)
    jbt = wf.get_jobs_run_by_time()
    wt = wf.get_wall_time()
    cwt = wf.get_cum_time()
    print("{} {} {}".format(jbt, wt, cwt))


def get_mean_results(data):
    temp = {}
    for k, v in data.items():
        # if k > 10:
        #     break
        # print(v)
        temp[k] = np.mean(v)
    print(temp)
    return temp


def main():
    # overhead_analysis(cluster_size=20)
    # Montage - Pegasus CS CN Analysis
    # plot_cs_analysis()
    plot_cn_analysis()
    # get_wf_results("/Users/rayson/Documents/master/pipe/submit/rayson/pegasus/pipeline/run0003")
    # plot_cs_cn_analysis()


if __name__ == '__main__':
    sys.exit(main())
