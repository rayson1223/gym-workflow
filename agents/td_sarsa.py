from gym import make
import gym_workflow.envs
from collections import defaultdict
import numpy as np
from agents.strategy.td import TD
import agents.utils.plotting as plt

if __name__ == '__main__':
	env = make('Montage-v4')
	episodes = 1000

	Q, stats = TD.sarsa(env, episodes)
	V = defaultdict(float)
	for state, action_values in Q.items():
		action_value = np.max(action_values)
		V[state] = action_value
	plt.plot_value_function(V, title="SARSA: Value Function representation - %s episodes" % episodes)
	print(Q)
	plt.plot_episode_stats(stats)
