from gym import make
import gym_workflow.envs
from collections import defaultdict, OrderedDict
import numpy as np
from agents.strategy.td import TD
import agents.utils.plotting as plt

if __name__ == '__main__':
    env = make('Montage-v8')
    episodes = 1000

    # Q Learning
    Q, stats, records = TD.q_learning(
        env, episodes, discount_factor=0.8, epsilon=0.3, training_episode=5, log_file="v8-training-epi-{}-vm-10.csv".format(episodes)
    )

    sQ = OrderedDict(sorted(Q.items()))
    plt.plot_simple_line(records["overhead"], xlabel="Episode", ylabel="Overhead(s)",
                        title="Episode vs Overhead(s)")
    plt.plot_line_value(sQ, title="Q-Learning: Value Function representation - %s episodes" % episodes)
    plt.plot_episode_stats(stats)
