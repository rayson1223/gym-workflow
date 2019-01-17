from gym_workflow.envs.montage_wf_env import MontageWfEnv
import random


def main():
    degree = 0.5
    for i in range(1001):
        cs = random.randint(1, 10)
        MontageWfEnv.run_experiment(cs=cs, cn=None, degrees=degree, file="workflow_record.csv")


if __name__ == "__main__":
    main()
