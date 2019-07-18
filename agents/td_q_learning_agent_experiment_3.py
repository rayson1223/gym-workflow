from gym import make
import gym_workflow.envs
from collections import defaultdict, OrderedDict
import numpy as np
from agents.strategy.td import TD
import agents.utils.plotting as plt

if __name__ == '__main__':
    env = make('Montage-v10')
    episodes = 50

    log_pre = "exp-3-epi-{}-train-0-maintain-all".format(episodes)
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

    plt.plot_exp_2_action_value(sQ, title="Experiment 3: Action Values Overview - %s episodes" % episodes)

    plt.overall_records_visualization(records['benchmark'], xlabel="Cycle", ylabel="Benchmark Makespan(s)",
                                      title="Experiment 3: Benchmark Makespan over Cycle ({} Episodes)".format(episodes))

    plt.overall_records_visualization(
        records['makespan'], xlabel='Cycle', ylabel="Makespan(s)",
        title="Experiment 3: Makespan(s) across Cycle ({} Episodes)".format(episodes)
    )

    # plt.v1_plot_action_value(sQ, title="Experiment 3: Action Values Overview - %s episodes" % episodes)

    # plt.v1_plot_episode_stats(stats)
    plt.plot_episode_stats(stats, smoothing_window=1)
