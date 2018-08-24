import numpy as np


class MontageWorkflowPolicyFactory:
	def __init__(self):
		self.__exec_time = []

	def make_random_action_policy(self, env):
		return env.action_space.sample()

	def make_epsilon_greedy_policy(self, Q, epsilon, nA):
		"""
			Creates an epsilon-greedy policy based on a given Q-function and epsilon.

			Args:
				Q: A dictionary that maps from state -> action-values.
						Each value is a numpy array of length nA (see below)
				epsilon: The probability to select a random action . float between 0 and 1.
				nA: Number of actions in the environment.

			Returns:
				A function that takes the observation as an argument and returns
				the probabilities for each action in the form of a numpy array of length nA.
		"""

		def policy_fn(observation):
			A = np.ones(nA, dtype=float) * epsilon / nA
			best_action = np.argmax(Q[observation])
			A[best_action] += (1.0 - epsilon)
			return A

		return policy_fn

	def create_greedy_policy(self, Q):
		"""
		Creates a greedy policy based on Q values.

		Args:
				Q: A dictionary that maps from state -> action values

		Returns:
				A function that takes an observation as input and returns a vector
				of action probabilities.
		"""

		def policy_fn(state):
			A = np.zeros_like(Q[state], dtype=float)
			best_action = np.argmax(Q[state])
			A[best_action] = 1.0
			return A

		return policy_fn

	def create_random_policy(self, nA):
		"""
		Creates a random policy function.

		Args:
				nA: Number of actions in the environment.

		Returns:
				A function that takes an observation as input and returns a vector
				of action probabilities
		"""
		A = np.ones(nA, dtype=float) / nA

		def policy_fn(observation):
			return A

		return policy_fn
