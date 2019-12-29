from gym_workflow.envs.montage_wf_env import MontageWfEnv


def main():
    degree = 6.0
    # for d in range(5):
    # 	degree += 0.1
    for cs in range(0, 31, 10):
        for i in range(2):
            if cs == 0:
                cs = 1
            MontageWfEnv.run_experiment(cs=cs + 1, cn=None, degrees=degree, file="testing_montage_scale_6.csv")


if __name__ == "__main__":
    main()
