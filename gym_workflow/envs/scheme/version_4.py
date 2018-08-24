from gym_workflow.envs.montage_wf_env import MontageWfEnv


class Version4(MontageWfEnv):
	"""
		@version 4.0:
			- Using Percentage as the value to gain reward.
			- 10% of the run will be the sample of experiment scales
	"""

	def step(self, action):
		assert self.action_space.contains(action)

		reward = 0.0
		self.last_action = action
		if action == 1:
			self.clusters_size += 1
		elif action == 2:
			self.clusters_size -= 1
		elif action == 3:
			self.clusters_num += 1
		elif action == 4:
			self.clusters_num -= 1

		# Range Guarding Function
		if self.clusters_size <= 0:
			reward -= 1.0
			self.clusters_size = 1
		elif self.clusters_size > 10:
			reward -= 1.0
			self.clusters_size = 10
		elif self.clusters_num <= 0:
			reward -= 1.0
			self.clusters_num = 1
		elif self.clusters_num > 10:
			reward -= 1.0
			self.clusters_num = 10
		else:
			res = self.run_gen_experiment(self.clusters_size, self.clusters_num)
			self.exec_time = res

			if self.best_exec_time is None:
				self.best_exec_time = res
			if self.last_exec_time is None:
				self.last_exec_time = res

		return self._get_obs(), reward, True, {}

	def render(self, mode='human'):
		outfile = StringIO() if mode == 'ansi' else sys.stdout
		init_msg = "Current Experiment Parameters: degree: %d\t cluster.size: %d\t cluster.num: %d\t\n" % (
			self.degree, self.clusters_size, self.clusters_num)
		outfile.write(init_msg)
		# if self.last_action is not None:
		# 	cs, cn = action
		result_str = "Current Execution Time: \t"
		expect_str = "Best Execution Time: \t"
		action_str = "Current Action: \t"
		# Process Outputs
		outfile.write(result_str + (" %s " % self.exec_time) + "\n")
		outfile.write(expect_str + (" %s " % self.best_exec_time) + "\n")
		outfile.write(action_str + (" %s " % self.last_action) + "\n")

		return outfile

	def reset(self):
		# Reset method should always return a new sets of episode settings
		# self.degree = 0.1
		# self.clusters_size = 1
		# self.clusters_num = 1
		# self.band_num = 1
		# self.best_exec_time = None
		if self.exec_time is not None:
			self.last_exec_time = self.exec_time
		self.wall_time = None
		self.cum_wall_time = None

		# print("Environment had been reset!")
		return self._get_obs()
