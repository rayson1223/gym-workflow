from gym import make
import gym_workflow.envs
from collections import defaultdict, OrderedDict
import numpy as np
from agents.strategy.td import TD
import agents.utils.plotting as plt

if __name__ == '__main__':
    env = make('Montage-v11')
    episodes = 100

    log_pre = "p2-exp1-epi-{}-train-0-maintain-all-terminal-20-M10-B20.csv".format(episodes)
    # log_pre = "exp-3-epi-{}-train-10-maintain-all".format(episodes)
    # log_pre = "exp-3-epi-{}-train-0-maintain-smallest".format(episodes)
    # log_pre = "exp-3-epi-{}-train-0-maintain-largest".format(episodes)
    #

    # Q Learning
    Q, stats, records = TD.q_learning(
        env, episodes, discount_factor=0.7, epsilon=0.3,
        training_episode=0, log_file=log_pre,
    )

    sQ = OrderedDict(sorted(Q.items()), key=lambda i: keyorder.index(i[0]))

    plt.plot_exp_2_action_value(sQ, title="Q-Value Overview - %s episodes" % episodes, opt_lower=8, opt_high=11, xlim=11, ylim=11)

    plt.plot_episode_stats(stats, smoothing_window=10)
