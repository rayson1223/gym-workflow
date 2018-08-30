from gym import make
import gym_workflow.envs
from collections import defaultdict
import numpy as np
from agents.strategy.monte_carlo import MonteCarlo
import agents.utils.plotting as plt

if __name__ == '__main__':
	env = make('Montage-v4')
	episodes = 1000

	Q, policy = MonteCarlo.mc_control_epsilon_greedy(env, num_episodes=episodes, epsilon=0.1)
	V = defaultdict(float)
	for state, actions in Q.items():
		action_value = np.max(actions)
		V[state] = action_value
	print(V)
	plt.plot_value_function(V, title="Greedy: Value Function representation - %s episodes" % episodes)
