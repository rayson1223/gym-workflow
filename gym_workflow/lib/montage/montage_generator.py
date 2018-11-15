from gym_workflow.envs.montage_wf_env import MontageWfEnv


def main():
	degree = 0
	for d in range(5):
		degree += 0.1
		for cs in range(10):
			for i in range(10):
				MontageWfEnv.run_experiment(cs=cs + 1, cn=None, degrees=degree, file="workflow_record_1.csv")


if __name__ == "__main__":
	main()
