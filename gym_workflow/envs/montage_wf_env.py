import gym
from gym.utils import seeding
from gym.spaces import Discrete, Tuple
from .database import DatabaseEnv
from io import StringIO
import sys
import random
from gym_workflow.lib.montage.montage import Montage


class WfEnv(gym.Env):
	# General Workflow Environment
	metadata = {'render.modes': ['human', 'ansi']}

	def __init__(self):
		# General Attributes where workflow is always capture
		self.exec_time = None
		self.wall_time = None
		self.cum_wall_time = None
		self._seed()

	def _seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]


class MontageWfEnv(WfEnv):
	"""
		As a workflow Env, it should always return [exec_time, wall_time, cumulative_wall_time] as the
		observations (known as [State] as well).

		As for this Montage Workflow Environment, the actions can be taken here are:
			- Increase cluster.size
			- Decrease cluster.size
			- Maintain cluster.size
	"""
	def __init__(self, degree=0.1, band_num=1, db_dir=".pegasus/workflow.db"):
		# Montage Experiment Variable
		super(MontageWfEnv, self).__init__()
		self.degree = degree
		self.clusters_size = 1
		self.clusters_num = 1
		self.is_clusters_size = True
		self.is_clusters_num = False
		self.band_num = band_num

		# Setting database connection
		self.db = DatabaseEnv(db_dir)

		self.action_space = Discrete(3)

		self.observation_space = Discrete(8), Discrete(8), Discrete(3)

		# Episode Conf
		self.best_exec_time = None
		self.last_exec_time = None
		self.last_action = None
		self.last_reward = None
		self.total_reward = 0.0
		# 0: Ntg, 1: improve, 2: degrade
		self.is_improve = 0
		self.seed()
		self.reset()

	def seed(self, seed=None):
		super()

	def step(self, action):
		"""
			Things to do:
				- Finalise the reward conditions
				- Current assuming all workflow will be successfully ran
			Possible Action
				- 0: Maintain
				- 1: Increase
				- 2: Decrease
		"""
		assert self.action_space.contains(action)
		cs = action

		def calc_lb_hb(v, p):
			return (v * (100 - p)) / 100, (v * (p + 100)) / 100

		reward = 0.0
		self.last_action = action
		if cs == 1:
			self.clusters_size += 1
		elif cs == 2:
			self.clusters_size -= 1

		if self.clusters_size <= 0 or self.clusters_size > 7:
			reward -= 1.0
			self.clusters_size = 1
		else:
			res = self.run_experiment()
			self.exec_time = res

			"""
				Reward Mechanism Manual:
					1) If it's better than the 'best record', it will be highly rewarded
					2) If it's perform better than LAST EXEC, it will be slightly rewarded
					3) If it's doesn't improve, no reward is given
					4) If it's perform worst than last exec, negative reward will be given
			"""
			res_lb, res_hb = calc_lb_hb(res, 10)

			if self.best_exec_time is None:
				self.best_exec_time = res
			if self.last_exec_time is None:
				self.last_exec_time = res

			# R1
			if res < self.best_exec_time:
				reward += 10.0
				self.is_improve = 1
			elif self.last_exec_time is not None:
				# R3
				lres_lb, lres_hb = calc_lb_hb(self.last_exec_time, 10)
				if lres_lb <= res <= lres_hb:
					self.is_improve = 0
					reward -= 1
				# R2 & 4
				elif res < self.last_exec_time:
					reward += 1.0
					self.is_improve = 1
				elif res > self.last_exec_time:
					reward -= 10.0
					self.is_improve = 2

		return self._get_obs(), reward, True, {}

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

		print("Environment had been reset!")
		return self._get_obs()

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

	# def _get_reward(self):
	#

	def _get_obs(self):
		return self.clusters_size, self.clusters_num

	# return self.clusters_size, self.clusters_num, self.is_improve

	def increase_level(self):
		# Determine whether should increase level of difficulty
		return None

	def run_experiment(self):
		montage = Montage()
		montage.build_transformation_catalog(self.clusters_size)
		# montage.generate_region_hdr()
		# montage.process_color_band()
		# montage.write_rc()
		# montage.write_property_conf()
		# montage.pegasus_plan()
		return montage.pegasus_run()
