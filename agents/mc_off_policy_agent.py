from gym import make
import gym_workflow.envs
from collections import defaultdict
import numpy as np
from agents.strategy.monte_carlo import MonteCarlo
from agents.policy.montage_workflow_policy_factory import MontageWorkflowPolicyFactory
import agents.utils.plotting as plt

if __name__ == '__main__':
	env = make('Montage-v4')
	episodes = 1000

	# MC Off Policy
	random_policy = MontageWorkflowPolicyFactory().create_random_policy(env.action_space.n)
	Q, policy = MonteCarlo.mc_control_importance_sampling(env, num_episodes=episodes, behavior_policy=random_policy)
	# For plotting: Create value function from action-value function
	# by picking the best action at each state
	V = defaultdict(float)
	for state, action_values in Q.items():
		action_value = np.max(action_values)
		V[state] = action_value

	plt.plot_value_function(V, title="Value Function representation - %s episodes" % episodes)
