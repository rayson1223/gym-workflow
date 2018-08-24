import gym
from gym.utils import seeding
from gym.spaces import Discrete, Tuple
from .database import DatabaseEnv
from io import StringIO
import sys
import random
import numpy as np
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
			Experiment Conditions:
				Independent Var:
					- Cluster Size
					- Cluster Number
					- Training Window Size [Applicable on version 4 onwards]

				Dependant Var:
					- Execution Time

				Constant Var:
					- No. of CPU
					- Assume all experiment are run successfully

			Env:
				Possible Action can be taken by agents
					- 0: Maintain
					- 1: Increase CS by 1
					- 2: Decrease CS by 1
					- 3: Increase CN by 1
					- 4: Decrease CN by 1

			Note:
				Things to decide:
					- Finalize reward functions
					- Examine Which Method is the best

				Questions to be answer:
					- Why used reinforcement learning rather than supervised / unsupervised learning
						- Because we want an adaptive learning agents and able to adapt in different situations

				Things to be proved:
					- Why the problem we are solving are regression problem but not linear problems (knn)?
						- Have to implement KNN and proved that the results unable to solve our problems
					- Has to proof that our learning method [Reinforcement Learning] are perform better than:
						- Random Forest
						- Extreme Gradient Boost [xgb]
						- Or any other regression algorithms
						- Compare in regression methodology where it's has a fair stand
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

		# @version 4.0 Config
		self.worst_exec_time = None
		self.exec_time_records = None

	def step(self, action):
		raise NotImplementedError

	def reset(self):
		raise NotImplementedError

	def render(self, mode='human'):
		raise NotImplementedError

	def _get_obs(self):
		return self.clusters_size, self.clusters_num

	def increase_level(self):
		# Determine whether should increase level of difficulty
		raise NotImplemented

	@staticmethod
	def run_static_experiment(cs, cn):
		return Montage.gen_static_exec_time(cs, cn)

	@staticmethod
	def run_gen_experiment(cs, cn):
		return Montage.gen_exec_time(cs, cn)

	def run_experiment(self):
		montage = Montage()
		montage.build_transformation_catalog(self.clusters_size, self.clusters_num)
		montage.generate_region_hdr()
		montage.process_color_band()
		montage.write_rc()
		montage.write_property_conf()
		montage.pegasus_plan()
