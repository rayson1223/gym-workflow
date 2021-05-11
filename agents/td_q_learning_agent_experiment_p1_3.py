from gym import make
import gym_workflow.envs
from collections import defaultdict, OrderedDict
import numpy as np
from agents.strategy.td import TD
import agents.utils.plotting as plt

if __name__ == '__main__':
    # Experiment abt proving up, down, maintain strategy failed

    env = make('Montage-v3')
    episodes = 100

    # Q Learning
    Q, stats, records = TD.q_learning(
        env, episodes, discount_factor=0.7, epsilon=0.3,
        training_episode=0, log_file="p1_3-training-0-epi-{}-vm-10.csv".format(episodes),
    )

    sQ = OrderedDict(sorted(Q.items()), key=lambda i:keyorder.index(i[0]))

    # plt.plot_exp_p1_3_action_value(sQ, title="Experiment 3: Q-Value Overview - %s episodes" % episodes)
    plt.v1_plot_action_value(sQ, title="Experiment 3: Q-Value Overview - %s episodes" % episodes)

    plt.v1_plot_episode_stats(stats)
