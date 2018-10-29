from gym import make
import gym_workflow.envs
from collections import defaultdict
import numpy as np
from agents.strategy.td import TD
import agents.utils.plotting as plt

if __name__ == '__main__':
	env = make('Montage-v5')
	episodes = 1000

	# Q Learning
	Q, stats = TD.q_learning(env, episodes, discount_factor=0.8)
	# print(Q)
	V = defaultdict(float)
	for state, action_values in Q.items():
		action_value = np.max(action_values)
		V[state] = action_value
	
	plt.plot_value_function(V, title="Q-Learning: Value Function representation - %s episodes" % episodes)
	# print(Q)
	plt.plot_episode_stats(stats)
