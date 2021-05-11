from gym import make
import gym_workflow.envs
from collections import defaultdict, OrderedDict
import numpy as np
from agents.strategy.td import TD
import agents.utils.plotting as plt

if __name__ == '__main__':
    env = make('Montage-v1')
    episodes = 100

    # Q Learning
    Q, stats, records = TD.q_learning(
        env, episodes, discount_factor=0.7, epsilon=0.3,
        training_episode=0, log_file="exp1-cn-training-0-epi-{}-vm-10.csv".format(episodes),
    )

    sQ = OrderedDict(sorted(Q.items()), key=lambda i:keyorder.index(i[0]))

    # plt.overhead_visualization(
    #     records['overhead'], xlabel='Cycle', ylabel="Overhead(s)",
    #     title="Overhead(s) across episodes"
    # )

    plt.v1_plot_action_value(sQ, title="Experiment 1: Q-Value Overview - %s episodes" % episodes)

    plt.v1_plot_episode_stats(stats)
