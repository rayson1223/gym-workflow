from gym_workflow.envs.montage_wf_env import MontageWfEnv
from gym_workflow.envs.database import DatabaseEnv
from gym.spaces import Discrete
import random
import sys


class Version4(MontageWfEnv):
	"""
		@version 4.0:
			- Using Percentage as the value to gain reward.
			- 10% of the run will be the sample of experiment scales
	"""
	
	def __init__(self):
		# Montage Experiment Variable
		super(Version4, self).__init__()
		self.degree = 0.1
		self.clusters_size = 1
		self.clusters_num = 1
		self.band_num = 1
		
		# Setting database connection
		self.db = None
		self.action_space = Discrete(5)
		self.observation_space = Discrete(8), Discrete(8), Discrete(3)
		
		# Episode Conf
		# Best exec_time: None or 1, depends on reward version
		self.best_exec_time = None
		self.worst_exec_time = None
		self.last_exec_time = None
		self.last_action = None
		self.reward = None
		self.total_reward = 0.0
		
		self.exec_records = []
		# self.gen_records(100)
		self.seed()
		self.reset()
	
	def configure(self, degree=0.1, band_num=1, db_dir=".pegasus/workflow.db", records=100):
		self.degree = degree
		self.band_num = band_num
		self.db = DatabaseEnv(db_dir)
		self.gen_records(records)
	
	def gen_records(self, episodes):
		if self.exec_records.__len__() > 0:
			self.exec_records = []
		for i in range(episodes):
			cs = random.randint(1, 5)
			cn = random.randint(1, 30)
			time = self.run_gen_experiment(cs, cn)
			self.exec_records.append(time)
		self.best_exec_time = min(self.exec_records)
		self.worst_exec_time = max(self.exec_records)
	
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
		elif self.clusters_num > 30:
			reward -= 1.0
			self.clusters_num = 30
		else:
			res = self.run_gen_experiment(self.clusters_size, self.clusters_num)
			self.exec_time = res
			
			if self.best_exec_time is None:
				self.best_exec_time = res
			if self.last_exec_time is None:
				self.last_exec_time = res
			
			improvement = (self.best_exec_time - self.exec_time) / self.best_exec_time * 100
			if improvement > 0:
				print(self.best_exec_time)
			if improvement > 0:
				reward = improvement
				self.best_exec_time = self.exec_time
			else:
				reward = -1
		self.reward = reward
		return self._get_obs(), reward, True, {}
	
	def render(self, mode='human'):
		outfile = StringIO() if mode == 'ansi' else sys.stdout
		init_msg = "Episodes Parameters: degree: %d\t cluster.size: %d\t cluster.num: %d\t\n" % (
			self.degree, self.clusters_size, self.clusters_num)
		outfile.write(init_msg)
		# if self.last_action is not None:
		# 	cs, cn = action
		result_str = "Current Execution Time: \t"
		expect_str = "Best Execution Time: \t"
		action_str = "Current Action: \t"
		reward_str = "Reward: \t"
		# Process Outputs
		outfile.write(result_str + (" %s " % self.exec_time) + "\n")
		outfile.write(expect_str + (" %s " % self.best_exec_time) + "\n")
		outfile.write(action_str + (" %s " % self.last_action) + "\n")
		outfile.write(reward_str + (" %s " % self.reward) + "\n")
		
		return outfile
	
	def reset(self):
		# print("Environment had been reset!")
		return self._get_obs()
	
	def _get_obs(self):
		return self.clusters_size, self.clusters_num
