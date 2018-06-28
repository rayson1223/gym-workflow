from gym import make
import gym_workflow.envs
from collections import defaultdict
import numpy as np
import sys
import matplotlib
import random

# matplotlib.use('ggplot')

if __name__ == '__main__':
	env = make('Montage-v0')
	last_exec = 0


	# Author: dennybritz
	# From URL: https://github.com/dennybritz/reinforcement-learning/blob/master/MC/MC%20Prediction%20Solution.ipynb
	def mc_prediction(policy, env, num_episodes, discount_factor=1.0):
		"""
			Monte Carlo prediction algorithm. Calculates the value function
			for a given policy using sampling.

			Args:
				policy: A function that maps an observation to action probabilities.
				env: OpenAI gym environment.
				num_episodes: Number of episodes to sample.
				discount_factor: Gamma discount factor.

			Returns:
				A dictionary that maps from state -> value.
				The state is a tuple and the value is a float.
		"""
		# Keeps track of sum and count of returns for each state
		# to calculate an average. We could use an array to save all
		# returns (like in the book) but that's memory inefficient.
		returns_sum = defaultdict(float)
		returns_count = defaultdict(float)

		# The final value function
		V = defaultdict(float)

		for i_episode in range(1, num_episodes + 1):
			# Print out which episode we're on, useful for debugging.
			if i_episode % 1000 == 0:
				print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
				sys.stdout.flush()

			# Generate an episode.
			# An episode is an array of (state, action, reward) tuples
			episode = []
			state = env.reset()
			for t in range(100):
				action = policy(state, env)
				next_state, reward, done, _ = env.step(action)
				episode.append((state, action, reward))
				if done:
					break
				state = next_state

			# Find all states the we've visited in this episode
			# We convert each state to a tuple so that we can use it as a dict key
			states_in_episode = set([tuple(x[0]) for x in episode])
			for state in states_in_episode:
				# Find the first occurance of the state in the episode
				first_occurence_idx = next(i for i, x in enumerate(episode) if x[0] == state)
				# Sum up all rewards since the first occurance
				G = sum([x[2] * (discount_factor ** i) for i, x in enumerate(episode[first_occurence_idx:])])
				# Calculate average return for this state over all sampled episodes
				returns_sum[state] += G
				returns_count[state] += 1.0
				V[state] = returns_sum[state] / returns_count[state]

		return V


	def calc_lb_hb(v, p):
		return (v * (100 - p)) / 100, (v * (p + 100)) / 100


	def wf_policy(observation, env):
		# Depend on the observation what actions I should be doing
		# cs, cn, im = observation

		return env.action_space.sample()


	def plot_value_function(V, title="Value Function"):
		"""
		Plots the value function as a surface plot.
		"""
		min_x = min(k[0] for k in V.keys())
		max_x = max(k[0] for k in V.keys())
		min_y = min(k[1] for k in V.keys())
		max_y = max(k[1] for k in V.keys())

		x_range = np.arange(min_x, max_x + 1)
		y_range = np.arange(min_y, max_y + 1)
		X, Y = np.meshgrid(x_range, y_range)

		# Find value for all (x, y) coordinates
		Z_noace = np.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y]))
		Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2, np.dstack([X, Y]))

		def plot_surface(X, Y, Z, title):
			fig = plt.figure(figsize=(20, 10))
			ax = fig.add_subplot(111, projection='3d')
			surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
			ax.set_xlabel('Player Sum')
			ax.set_ylabel('Dealer Showing')
			ax.set_zlabel('Value')
			ax.set_title(title)
			ax.view_init(ax.elev, -120)
			fig.colorbar(surf)
			plt.show()

		plot_surface(X, Y, Z_noace, "{} (No Usable Ace)".format(title))
		plot_surface(X, Y, Z_ace, "{} (Usable Ace)".format(title))


	V_10k = mc_prediction(wf_policy, env, num_episodes=1000)
	print(V_10k)
# plot_value_function(V_10k, title="10 Steps")
