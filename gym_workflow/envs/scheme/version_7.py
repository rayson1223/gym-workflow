from gym_workflow.envs.montage_wf_env import MontageWfEnv
from gym_workflow.envs.database import DatabaseEnv
from gym.spaces import Discrete
import random
import numpy as np
import sys
from gym_workflow.libs.recording import *


class Version7(MontageWfEnv):
	"""
		@version 7.0:
			- Remove cluster numbers, focus cluster size
			- Update from episodic to continuous RL
			- For every cluster size, it will have it own min-max values for verify how good it perform
			- Remove the remain actions, remain only up and down
			
			
	"""
	
	def __init__(self):
		CLUSTER_RANGE = 10
		ACTION_RANGE = 3
		# Montage Experiment Variable
		super(Version7, self).__init__()
		self.degree = 0.5
		self.clusters_size = 1
		self.clusters_num = None
		self.band_num = 1
		
		# Setting database connection
		self.db = None
		self.action_space = Discrete(ACTION_RANGE)
		self.observation_space = Discrete(CLUSTER_RANGE), Discrete(ACTION_RANGE)
		
		# Episode Conf
		# Best exec_time: None or 1, depends on reward version
		self.best_exec_time = None
		self.last_exec_time = None
		self.last_action = None
		self.reward = 0
		self.total_reward = 0.0
		
		self.exec_records = {}
		self.all_exec_record = list()
		self.seed()
		self.reset()
	
	def configure(self, degree=0.1, band_num=1, db_dir=".pegasus/workflow.db", records=100):
		self.degree = degree
		self.band_num = band_num
		self.db = DatabaseEnv(db_dir)
	
	def step(self, action):
		assert self.action_space.contains(action)
		done = True
		reward = 0.0
		
		self.last_action = action
		if action == 0:
			self.clusters_size += 1
		elif action == 1:
			self.clusters_size -= 1
		
		# Range Guarding Function
		if self.clusters_size <= 0:
			self.clusters_size = 1
			reward = -100
			write_episode(
				[self._get_obs(), action, None, None, None, None, self.best_exec_time, None, reward],
				file_name="v7_workflow_record.csv"
			)
		elif self.clusters_size > 10:
			self.clusters_size = 10
			reward = -100
			write_episode(
				[self._get_obs(), action, None, None, None, None, self.best_exec_time, None, reward],
				file_name="v7_workflow_record.csv"
			)
		else:
			# Return all the data collected
			# status, jb, wt, cwt = self.run_experiment(cs=self.clusters_size, degrees=0.5)
			result = self.run_cs_gen_experiment(self.clusters_size)
			
			# Experiment run failed -> High Penalty
			# if not status:
			# 	return self._get_obs(), -10, True, {}
			#
			self.exec_time = result
			
			# Fine Tune Records set within the cluster size
			if self.clusters_size in self.exec_records:
				if self.exec_time > self.exec_records[self.clusters_size]['max']:
					self.exec_records[self.clusters_size]['max'] = self.exec_time
				elif self.exec_time < self.exec_records[self.clusters_size]['min']:
					self.exec_records[self.clusters_size]['min'] = self.exec_time
			else:
				self.exec_records[self.clusters_size] = {
					'min': 0,
					'max': 0
				}
			self.all_exec_record.append(self.exec_time)
			# Setting up best exec time
			if self.best_exec_time is None:
				self.best_exec_time = self.exec_time
			if self.last_exec_time is None:
				self.last_exec_time = self.exec_time
			
			# Rewarding / Penalty Judgement
			if self.exec_time < np.percentile(self.all_exec_record, 20):
				self.best_exec_time = self.exec_time
				reward = 500
			elif self.exec_time > self.last_exec_time:
				reward = -100
			elif self.exec_time <= self.last_exec_time:
				reward = -10
			self.last_exec_time = self.exec_time
			
			# write_episode(
			# 	[self._get_obs(), action, status, jb, wt, cwt, self.best_exec_time, None, reward],
			# 	file_name="v7_workflow_record.csv"
			# )
		self.total_reward += reward
		
		return self._get_obs(), reward, done, {}
	
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
		self.clusters_size = random.randint(1, 10)
		# self.clusters_num = random.randint(1, 10)
		return self.clusters_size  # , self.clusters_num
	
	def _get_obs(self):
		return self.clusters_size  # , self.clusters_num
