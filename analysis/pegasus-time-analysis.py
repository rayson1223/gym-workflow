import sys
import csv
from functools import reduce


def main():
    # Workflow with cn=30, id= /home/rayson/gym-workflow/gym_workflow/libs/montage/work/1567591628199
    data = {}
    total_runtime = []
    total_seqexec = []
    total_seqexec_delay = []
    total_condorq = []
    total_post = []
    with open('./data/fundamental/split-slp-jobs.txt') as f:
        next(f)
        for line in f:
            # print(line)
            test = list(filter(None, line.replace('-', '0').split(' ')))
            # print(line.replace('-', '0'))
            if test[0] == '#' or test[0] == 'Job':
                continue
            data[test[0]] = {
                "try": int(test[1]),
                "site": test[2],
                "kickstart": float(test[3]),
                "mult": int(test[4]),
                "kickstart-mult": float(test[5]),
                "cpu-time": test[6],
                "post": float(test[7]),
                "condor_q_time": float(test[8]),
                "resource": test[9],
                "runtime": float(test[10]),
                "seqexec": float(test[11]),
                "seqexec_delay": float(test[12]),
                "exitcode": test[13],
                "hostname": test[14]
            }
            total_runtime.append(data[test[0]]["runtime"])
            total_seqexec.append(data[test[0]]['seqexec'])
            total_seqexec_delay.append(data[test[0]]['seqexec_delay'])
            total_condorq.append(data[test[0]]['condor_q_time'])
            total_post.append(data[test[0]]['post'])
    print("Total SeqExec Time: {} \n".format(sum(total_seqexec)))
    print("Total SeqExec Delay Time: {} \n".format(sum(total_seqexec_delay)))
    print("Total Condor Q Time: {} \n".format(sum(total_condorq)))
    print("Total Run Time: {} \n".format(sum(total_runtime)))
    print("Total Post: {} \n".format(sum(total_post)))
    print("Total Addup: {}\n".format(
        reduce(
            (lambda x, y: x + y),
            [sum(total_seqexec), sum(total_seqexec_delay), sum(total_condorq), sum(total_runtime), sum(total_post)]
        )))


if __name__ == '__main__':
    sys.exit(main())
