import sys
from collections import defaultdict
import itertools
from gym import make
import gym_workflow.envs
import agents.utils.plotting as plt
from collections import namedtuple
import numpy as np

if __name__ == '__main__':
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

		# The publication value function
		V = defaultdict(float)

		# EpisodeStats = namedtuple("Stats", ["episode_states", "episode_actions", "episode_rewards"])
		# stats = EpisodeStats(
		# 	episode_states=np.zeros(num_episodes),
		# 	episode_actions=np.zeros(num_episodes),
		# 	episode_rewards=np.zeros(num_episodes)
		# )

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
				episode.append((next_state, action, reward))
				# stats.episode_states[i_episode] = state
				# stats.episode_actions[i_episode] = action
				# stats.episode_rewards[i_episode] = reward
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


	def wf_policy(observation, env):
		# Depend on the observation what actions I should be doing
		# cs, cn, im = observation

		return env.action_space.sample()
	
	
	env = make('Montage-v3')
	episodes = 10000

	V_10k = mc_prediction(wf_policy, env, num_episodes=episodes)
	print(V_10k)
	
	plt.plot_value_function(V_10k, title="Monte Carlo First Visit with {} episodes".format(100000))
