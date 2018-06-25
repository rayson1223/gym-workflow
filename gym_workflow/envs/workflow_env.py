import gym
from gym.spaces import Discrete, Tuple
from .database import DatabaseEnv
from io import StringIO
import sys
import random
from gym_workflow.lib.montage.montage import Montage


class WorkflowEnv(gym.Env):
	metadata = {'render.modes': ['human', 'ansi']}

	def __init__(self, degree=0.1, band_num=1, db_dir=".pegasus/workflow.db"):
		# Montage Experiment Variable
		self.degree = degree
		self.clusters_size = 1
		self.clusters_num = 1
		self.is_clusters_size = True
		self.is_clusters_num = False
		self.band_num = band_num

		# Setting database connection
		self.db = DatabaseEnv(db_dir)

		# Sub-actions list [clusters.size, cluster.num]:
		#   - No Change: 0
		#   - Increase by 1: 1
		#   - Decrease by 1: 2
		self.action_space = Discrete(3)

		self.observation_space = Discrete(3)

		# Episode Conf
		self.best_exec_time = None
		self.last_exec_time = None
		self.last_action = None
		self.last_reward = None
		self.total_reward = 0.0
		self.seed()
		self.reset()

	def step(self, action):
		assert self.action_space.contains(action)
		cs = action

		# TODO: Determine termination conditions
		# For now:
		#   1) Experiment run failed
		#   2) Experiment all jobs is done but is not updated
		#   3) No matter how many time increase the cluster size,
		#       the execution time is almost the same
		done = False
		reward = 0.0
		self.last_action = action
		if cs == 1:
			self.clusters_size -= 1
		elif cs == 2:
			self.clusters_size += 1

		if self.clusters_size <= 0 or self.clusters_size > 7:
			reward -= 1.0
			self.clusters_size = 1
		else:
			res = self.run_experiment()
			self.last_exec_time = res
			# Determine the reward mechanism
			if self.best_exec_time == None:
				self.best_exec_time = res
			# reward += 1.0
			# if last_best_exec_time-5 <= res <= last_best_exec_time+5
			elif (self.best_exec_time - 10) <= res <= (self.best_exec_time + 10):
				if res < self.best_exec_time:
					self.best_exec_time = res
					reward += 1.0
				else:
					reward += 0.5
			elif res < (self.best_exec_time - 10):
				self.best_exec_time = res
				reward += 1.0
			else:
				reward -= 0.5

		self.last_reward = reward
		self.total_reward += reward

		if self.total_reward >= 5.0 or self.total_reward < -5.0:
			done = True

		return action, reward, done, self.total_reward

	def reset(self):
		self.degree = 0.1
		self.clusters_size = 1
		self.clusters_num = 1
		self.band_num = 1
		self.best_exec_time = None
		self.last_exec_time = None
		self.last_action = None
		self.last_reward = None
		self.total_reward = 0.0

		print("Environment had been reset!")

	def render(self, mode='human'):
		outfile = StringIO() if mode == 'ansi' else sys.stdout
		init_msg = "Current Experiment Parameters: degree: %d\t cluster.size: %d\t cluster.num: %d\t\n" % (
			self.degree, self.clusters_size, self.clusters_num)
		outfile.write(init_msg)
		# if self.last_action is not None:
		# 	cs, cn = action
		result_str = "Workflow Execution Time: \t"
		expect_str = "Expected Execution Time: \t"

		# Process Outputs
		outfile.write(result_str + (" %s " % self.last_exec_time) + "\n")
		outfile.write(expect_str + (" %s " % self.best_exec_time) + "\n")

		return outfile

	def increase_level(self):
		# Determine whether should increase level of difficulty
		return None

	def run_experiment(self):
		montage = Montage()
		# montage.build_transformation_catalog(self.clusters_size)
		# montage.generate_region_hdr()
		# montage.process_color_band()
		# montage.write_rc()
		# montage.write_property_conf()
		# montage.pegasus_plan()
		return montage.pegasus_run()
