from gym_workflow.envs.montage_wf_env import MontageWfEnv
from gym_workflow.envs.database import DatabaseEnv
from gym.spaces import Discrete
import random
import sys
from gym_workflow.lib.recording import *


class Version6(MontageWfEnv):
	"""
		@version 6.0:
			- Redefined episodes terminal state
			
			Terminal State:
				- cluster size, 0 < cs <= 10, any other number terminate
				- cluster num, 0 < cn <= 10, any other number terminate
				- episode length is greater than 100
				- accumulate reward > 200
	"""
	
	def __init__(self):
		CLUSTER_RANGE = 10
		ACTION_RANGE = 5
		# Montage Experiment Variable
		super(Version6, self).__init__()
		self.degree = 0.1
		self.clusters_size = 1
		self.clusters_num = 1
		self.band_num = 1
		
		# Setting database connection
		self.db = None
		self.action_space = Discrete(ACTION_RANGE)
		self.observation_space = Discrete(CLUSTER_RANGE), Discrete(CLUSTER_RANGE), Discrete(ACTION_RANGE)
		
		# Episode Conf
		# Best exec_time: None or 1, depends on reward version
		self.best_exec_time = None
		self.worst_exec_time = None
		self.last_exec_time = None
		self.last_action = None
		self.reward = 0
		self.total_reward = 0.0
		
		self.terminate_count = 0
		
		self.exec_records = []
		self.seed()
		self.reset()
	
	def configure(self, degree=0.1, band_num=1, db_dir=".pegasus/workflow.db", records=100):
		self.degree = degree
		self.band_num = band_num
		self.db = DatabaseEnv(db_dir)
	
	def step(self, action):
		assert self.action_space.contains(action)
		done = False
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
			reward = -1.0
			self.clusters_size = 1
			done = True
		# write_episode([self._get_obs(), action, None, None, None, None, self.best_exec_time, None, reward])
		elif self.clusters_size > 10:
			reward = -1.0
			self.clusters_size = 10
			done = True
		# write_episode([self._get_obs(), action, None, None, None, None, self.best_exec_time, None, reward])
		elif self.clusters_num <= 0:
			reward = -1.0
			self.clusters_num = 1
			done = True
		# write_episode([self._get_obs(), action, None, None, None, None, self.best_exec_time, None, reward])
		elif self.clusters_num > 10:
			reward = -1.0
			self.clusters_num = 10
			done = True
		# write_episode([self._get_obs(), action, None, None, None, None, self.best_exec_time, None, reward])
		else:
			# Return all the data collected
			# status, jb, wt, cwt = self.run_experiment(self.clusters_size, self.clusters_num)
			result = self.run_gen_experiment(self.clusters_size, self.clusters_num)
			
			# Experiment run failed -> High Penalty
			# if not status:
			# 	return self._get_obs(), -10, True, {}
			#
			self.exec_time = result
			
			if self.best_exec_time is None:
				self.best_exec_time = self.exec_time
			if self.last_exec_time is None:
				self.last_exec_time = self.exec_time
			
			def calc_lb_hb(v, p):
				return (v * (100 - p)) / 100, (v * (p + 100)) / 100
			
			lres_lb, lres_hb = calc_lb_hb(self.last_exec_time, 10)
			if self.exec_time <= lres_lb and self.exec_time <= self.best_exec_time:
				reward = 20
			elif lres_lb >= self.exec_time >= self.best_exec_time:
				reward = 10
			elif lres_lb <= self.exec_time <= lres_hb:
				reward = -1
			else:
				reward = -20
		
		# write_episode([self._get_obs(), action, status, jb, wt, cwt, self.best_exec_time, improvement, reward])
		self.reward += reward
		if self.reward >= 200:
			done = True
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
		self.terminate_count = 0
		self.reward = 0
		self.clusters_size = random.randint(1, 10)
		self.clusters_num = random.randint(1, 10)
		return self.clusters_size, self.clusters_num
	
	def _get_obs(self):
		return self.clusters_size, self.clusters_num
