import gym
from gym.utils import seeding
from gym_workflow.lib.montage.montage import Montage
import time


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
	
	def step(self, action):
		pass
	
	def reset(self):
		pass
	
	def render(self, mode='human'):
		pass


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
	
	def step(self, action):
		raise NotImplementedError
	
	def reset(self):
		raise NotImplementedError
	
	def render(self, mode='human'):
		raise NotImplementedError
	
	def _get_obs(self):
		raise NotImplemented
	
	def increase_level(self):
		# Determine whether should increase level of difficulty
		raise NotImplemented
	
	@staticmethod
	def run_static_experiment(cs, cn):
		return Montage.gen_static_exec_time(cs, cn)
	
	@staticmethod
	def run_gen_experiment(cs, cn):
		return Montage.gen_exec_time(cs, cn)
	
	@staticmethod
	def run_experiment(cs=1, cn=None, degrees=0.1, band=["2mass:j:red"], file="workflow_record.csv"):
		montage = Montage(degrees=degrees, band=band)
		montage.build(cs, cn)
		montage.pegasus_run()
		# Wait for the job submission status
		time.sleep(5)
		# Wait for the results
		montage.write_record(cs, cn, filename=file)
		complete = montage.monitor_experiment_completion()
		
		if complete:
			jb, wt, cwt = montage.get_results()
			return True, jb, wt, cwt
		else:
			return False, None, None, None
