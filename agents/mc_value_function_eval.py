from gym import make
import gym_workflow.envs
from collections import defaultdict
import numpy as np
import pandas as pd
import sys
from collections import namedtuple
import random
import itertools
import agents.utils.plotting as plt
from agents.policy.montage_workflow_policy_factory import MontageWorkflowPolicyFactory


class MonteCarlo:
	@staticmethod
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
				action = policy(state)
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

	@staticmethod
	def mc_control_epsilon_greedy(env, num_episodes, discount_factor=1.0, epsilon=0.1):
		"""
		Monte Carlo Control using Epsilon-Greedy policies.
		Finds an optimal epsilon-greedy policy.

		Args:
				env: OpenAI gym environment.
				num_episodes: Number of episodes to sample.
				discount_factor: Gamma discount factor.
				epsilon: Chance the sample a random action. Float betwen 0 and 1.

		Returns:
				A tuple (Q, policy).
				Q is a dictionary mapping state -> action values.
				policy is a function that takes an observation as an argument and returns
				action probabilities
		"""

		# Keeps track of sum and count of returns for each state
		# to calculate an average. We could use an array to save all
		# returns (like in the book) but that's memory inefficient.
		returns_sum = defaultdict(float)
		returns_count = defaultdict(float)

		# The final action-value function.
		# A nested dictionary that maps state -> (action -> action-value).
		Q = defaultdict(lambda: np.zeros(env.action_space.n))

		# The policy we're following
		policy = MontageWorkflowPolicyFactory().make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
		# policy = MontageWorkflowPolicyFactory().create_random_policy(env.action_space.n)

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
				probs = policy(state)
				action = np.random.choice(np.arange(len(probs)), p=probs)
				next_state, reward, done, _ = env.step(action)
				episode.append((state, action, reward))
				# print("Episode: %s" % i_episode)
				# env.render()
				# print("Policy: %s" % probs)
				# print("Reward: %s \n\n" % reward)

				if done:
					break

			# Find all (state, action) pairs we've visited in this episode
			# We convert each state to a tuple so that we can use it as a dict key
			sa_in_episode = set([(tuple(x[0]), x[1]) for x in episode])
			for state, action in sa_in_episode:
				sa_pair = (state, action)
				# Find the first occurance of the (state, action) pair in the episode
				first_occurence_idx = next(i for i, x in enumerate(episode) if x[0] == state and x[1] == action)
				# Sum up all rewards since the first occurance
				G = sum([x[2] * (discount_factor ** i) for i, x in enumerate(episode[first_occurence_idx:])])
				# Calculate average return for this state over all sampled episodes
				returns_sum[sa_pair] += G
				returns_count[sa_pair] += 1.0
				Q[state][action] = returns_sum[sa_pair] / returns_count[sa_pair]

		# The policy is improved implicitly by changing the Q dictionary
		# print(returns_count)
		return Q, policy

	@staticmethod
	def mc_control_importance_sampling(env, num_episodes, behavior_policy, discount_factor=1.0):
		"""
		Monte Carlo Control Off-Policy Control using Weighted Importance Sampling.
		Finds an optimal greedy policy.

		Args:
				env: OpenAI gym environment.
				num_episodes: Number of episodes to sample.
				behavior_policy: The behavior to follow while generating episodes.
						A function that given an observation returns a vector of probabilities for each action.
				discount_factor: Gamma discount factor.

		Returns:
				A tuple (Q, policy).
				Q is a dictionary mapping state -> action values.
				policy is a function that takes an observation as an argument and returns
				action probabilities. This is the optimal greedy policy.
		"""

		# The final action-value function.
		# A dictionary that maps state -> action values
		Q = defaultdict(lambda: np.zeros(env.action_space.n))
		# The cumulative denominator of the weighted importance sampling formula
		# (across all episodes)
		C = defaultdict(lambda: np.zeros(env.action_space.n))

		# Our greedily policy we want to learn
		target_policy = MontageWorkflowPolicyFactory().create_greedy_policy(Q)

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
				# Sample an action from our policy
				probs = behavior_policy(state)
				action = np.random.choice(np.arange(len(probs)), p=probs)
				next_state, reward, done, _ = env.step(action)
				episode.append((state, action, reward))
				if done:
					break
				state = next_state

			# Sum of discounted returns
			G = 0.0
			# The importance sampling ratio (the weights of the returns)
			W = 1.0
			# For each step in the episode, backwards
			for t in range(len(episode))[::-1]:
				state, action, reward = episode[t]
				# Update the total reward since step t
				G = discount_factor * G + reward
				# Update weighted importance sampling formula denominator
				C[state][action] += W
				# Update the action-value function using the incremental update formula (5.7)
				# This also improves our target policy which holds a reference to Q
				Q[state][action] += (W / C[state][action]) * (G - Q[state][action])
				# If the action taken by the behavior policy is not the action
				# taken by the target policy the probability will be 0 and we can break
				if action != np.argmax(target_policy(state)):
					break
				W = W * 1. / behavior_policy(state)[action]

		return Q, target_policy


class TD:

	@staticmethod
	def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
		"""
		Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
		while following an epsilon-greedy policy

		Args:
				env: OpenAI environment.
				num_episodes: Number of episodes to run for.
				discount_factor: Gamma discount factor.
				alpha: TD learning rate.
				epsilon: Chance the sample a random action. Float betwen 0 and 1.

		Returns:
				A tuple (Q, episode_lengths).
				Q is the optimal action-value function, a dictionary mapping state -> action values.
				stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
		"""
		EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])
		# The final action-value function.
		# A nested dictionary that maps state -> (action -> action-value).
		Q = defaultdict(lambda: np.zeros(env.action_space.n))

		# Keeps track of useful statistics
		stats = EpisodeStats(
			episode_lengths=np.zeros(num_episodes),
			episode_rewards=np.zeros(num_episodes))

		# The policy we're following
		policy = MontageWorkflowPolicyFactory().make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

		for i_episode in range(num_episodes):
			# Print out which episode we're on, useful for debugging.
			if (i_episode + 1) % 100 == 0:
				print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
				sys.stdout.flush()

			# Reset the environment and pick the first action
			state = env.reset()

			# One step in the environment
			# total_reward = 0.0
			for t in itertools.count():

				# Take a step
				action_probs = policy(state)
				action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
				next_state, reward, done, _ = env.step(action)

				# Update statistics
				stats.episode_rewards[i_episode] += reward
				stats.episode_lengths[i_episode] = t

				# TD Update
				best_next_action = np.argmax(Q[next_state])
				td_target = reward + discount_factor * Q[next_state][best_next_action]
				td_delta = td_target - Q[state][action]
				Q[state][action] += alpha * td_delta

				if done:
					break

				state = next_state

		return Q, stats

	@staticmethod
	def sarsa(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
		"""
		SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.

		Args:
				env: OpenAI environment.
				num_episodes: Number of episodes to run for.
				discount_factor: Gamma discount factor.
				alpha: TD learning rate.
				epsilon: Chance the sample a random action. Float betwen 0 and 1.

		Returns:
				A tuple (Q, stats).
				Q is the optimal action-value function, a dictionary mapping state -> action values.
				stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
		"""

		# The final action-value function.
		# A nested dictionary that maps state -> (action -> action-value).
		Q = defaultdict(lambda: np.zeros(env.action_space.n))
		EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])

		# Keeps track of useful statistics
		stats = EpisodeStats(
			episode_lengths=np.zeros(num_episodes),
			episode_rewards=np.zeros(num_episodes))

		# The policy we're following
		policy = MontageWorkflowPolicyFactory().make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

		for i_episode in range(num_episodes):
			# Print out which episode we're on, useful for debugging.
			if (i_episode + 1) % 100 == 0:
				print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
				sys.stdout.flush()

			# Reset the environment and pick the first action
			state = env.reset()
			action_probs = policy(state)
			action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

			# One step in the environment
			for t in itertools.count():
				# Take a step
				next_state, reward, done, _ = env.step(action)

				# Pick the next action
				next_action_probs = policy(next_state)
				next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)

				# Update statistics
				stats.episode_rewards[i_episode] += reward
				stats.episode_lengths[i_episode] = t

				# TD Update
				td_target = reward + discount_factor * Q[next_state][next_action]
				td_delta = td_target - Q[state][action]
				Q[state][action] += alpha * td_delta

				if done:
					break

				action = next_action
				state = next_state

		return Q, stats


if __name__ == '__main__':
	env = make('Montage-v0')
	episodes = 1000

	# Q, policy = MonteCarlo.mc_control_epsilon_greedy(env, num_episodes=episodes, epsilon=0.1)
	# V = defaultdict(float)
	# for state, actions in Q.items():
	# 	action_value = np.max(actions)
	# 	V[state] = action_value
	# print(V)
	# plot_value_function(V, title="Greedy: Value Function representation - %s episodes" % episodes)

	# MC Off Policy
	# random_policy = MontageWorkflowPolicyFactory().create_random_policy(env.action_space.n)
	# Q, policy = MonteCarlo.mc_control_importance_sampling(env, num_episodes=episodes, behavior_policy=random_policy)
	# # For plotting: Create value function from action-value function
	# # by picking the best action at each state
	# V = defaultdict(float)
	# for state, action_values in Q.items():
	# 	action_value = np.max(action_values)
	# 	V[state] = action_value
	#
	# plot_value_function(V, title="Value Function representation - %s episodes" % episodes)

	# Q Learning
	Q, stats = TD.q_learning(env, episodes)
	V = defaultdict(float)
	for state, action_values in Q.items():
		action_value = np.max(action_values)
		V[state] = action_value
	plt.plot_value_function(V, title="Q-Learning: Value Function representation - %s episodes" % episodes)
	# print(Q)
	plt.plot_episode_stats(stats)

# Q, stats = TD.sarsa(env, episodes)
# V = defaultdict(float)
# for state, action_values in Q.items():
# 	action_value = np.max(action_values)
# 	V[state] = action_value
# plot_value_function(V, title="SARSA: Value Function representation - %s episodes" % episodes)
# print(Q)
# plot_episode_stats(stats)
