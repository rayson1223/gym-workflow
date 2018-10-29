from gym_workflow.envs.montage_wf_env import MontageWfEnv
from gym_workflow.envs.database import DatabaseEnv
from gym.spaces import Discrete, Tuple
import numpy as np
import random

class Version3(MontageWfEnv):
	"""
		@version 3.0:
			Adopt reward mechanism with activation functions,
			if the best exec time doesn't exec the trigger, it will not getting any reward

		Conclusion:
			- Success provided the range of the results are similar and distinct as set in static exec time
			- Failed if is random gen range of exec time
	"""

	def __init__(self, degree=0.1, band_num=1, db_dir=".pegasus/workflow.db"):
		# Montage Experiment Variable
		super(Version3, self).__init__()
		self.degree = degree
		self.clusters_size = 1
		self.clusters_num = 1
		self.is_clusters_size = True
		self.is_clusters_num = False
		self.band_num = band_num

		# Setting database connection
		self.db = DatabaseEnv(db_dir)

		self.action_space = Discrete(5)

		self.observation_space = Discrete(8), Discrete(8), Discrete(3)

		# Episode Conf
		# Best exec_time: None or 1, depends on reward version
		self.best_exec_time = None
		self.last_exec_time = None
		self.last_action = None
		self.last_reward = None
		self.total_reward = 0.0
		# 0: Ntg, 1: improve, 2: degrade
		self.is_improve = 0
		self.seed()
		self.reset()

	def step(self, action):
		assert self.action_space.contains(action)

		def calc_lb_hb(v, p):
			return (v * (100 - p)) / 100, (v * (p + 100)) / 100

		reward = 0.0
		self.last_action = action
		done = False
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

			if self.best_exec_time is None:
				self.best_exec_time = self.exec_time
			elif (
				self.best_exec_time - self.exec_time) / self.best_exec_time > 0.6 or self.best_exec_time == self.exec_time:
				self.best_exec_time = self.exec_time
				reward = 10
			else:
				reward = -1
		if self.total_reward > 50:
			done = True
		return self._get_obs(), reward, done, {}

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
		self.total_reward = 0
		self.clusters_size = random.randint(1, 10)
		self.clusters_num = random.randint(1, 10)
		
		# print("Environment had been reset!")
		return self.clusters_size, self.clusters_num
	
	def _get_obs(self):
		return self.clusters_size, self.clusters_num
