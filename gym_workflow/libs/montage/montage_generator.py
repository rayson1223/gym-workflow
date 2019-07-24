from gym_workflow.envs.montage_wf_env import MontageWfEnv


def main():
	degree = 0.5
	# for d in range(5):
	# 	degree += 0.1
	for cs in range(30):
		for i in range(30):
			MontageWfEnv.run_experiment(cs=cs + 1, cn=None, degrees=degree, file="cluster_size_analysis_montage_05.csv")


if __name__ == "__main__":
	main()
