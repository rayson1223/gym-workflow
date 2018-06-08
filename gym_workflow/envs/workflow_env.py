import gym
from gym import error, spaces, utils
from gym.utils import seeding
from .database import DatabaseEnv

class WorkflowEnv(gym.Env):
	metadata = {'render.modes': ['human', 'ansi']}

	def __init__(self, degree=0.1, band_num=1, db_dir="~/.pegasus/workflow.db"):
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
		self.action_space = Tuple([
			spaces.Discrete(3), spaces.Discrete(3)
		])
		self.observation_space = [self.degree, self.clusters_size, self.clusters_num, self.band_num]

		self.seed()
		self.reset()

	def step(self, action):
		assert self.action_space.contains(action)

	def reset(self):
		self.degree = 0.1
		self.clusters_size = 1
		self.clusters_num = 1
		self.band_num = 1

	def render(self, mode='human'):
		outfile = StringIO() if mode == 'ansi' else sys.stdout
		init_msg = "Current Experiment Parameters:\n degree: %d\t cluster.size: %d\t cluster.num: %d\t\n" % (self.degree, self.clusters_size, self.clusters_num)
		outfile.write(init_msg)
