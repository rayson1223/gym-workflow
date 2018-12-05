from gym import make
import gym_workflow.envs
from collections import defaultdict, OrderedDict
import numpy as np
from agents.strategy.td import TD
import agents.utils.plotting as plt

if __name__ == '__main__':
	env = make('Montage-v7')
	episodes = 1000

	# Q Learning
	Q, stats = TD.q_learning(env, episodes, discount_factor=0.8, epsilon=0.3)
	
	sQ = OrderedDict(sorted(Q.items()))
	
	plt.plot_line_value(sQ, title="Q-Learning: Value Function representation - %s episodes" % episodes)
	plt.plot_episode_stats(stats)
