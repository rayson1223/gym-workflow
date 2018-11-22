import numpy as np
import itertools
import sys
from collections import defaultdict
from collections import namedtuple
from agents.policy.montage_workflow_policy_factory import MontageWorkflowPolicyFactory
from gym_workflow.lib.recording import *
import agents.utils.plotting as plt


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
				# V = defaultdict(float)
				# for state, action_values in Q.items():
				# 	action_value = np.max(action_values)
				# 	V[state] = action_value
				# plt.plot_value_function(V, title="Q-Learning: Value Function representation - %s episodes" % (
				# 		i_episode + 1))
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
				# print("Count %s: %s, State: %s,  Reward: %s" % (t, done, next_state, reward))
				if done or t > 100:
					# write_training_status([i_episode, Q, stats, action, action_probs, reward])
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
