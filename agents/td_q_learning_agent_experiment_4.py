from gym import make
import gym_workflow.envs
from collections import defaultdict, OrderedDict
import numpy as np
from agents.strategy.td import TD
import agents.utils.plotting as plt

if __name__ == '__main__':
    env = make('Montage-v8')
    episodes = 200

    # Q Learning
    Q, stats, records = TD.q_learning(
        env, episodes, discount_factor=0.7, epsilon=0.3,
        training_episode=0, log_file="v8-training-epi-{}-vm-100.csv".format(episodes),
    )

    sQ = OrderedDict(sorted(Q.items()))
    plt.plot_exp_3_action_value(sQ, title="exp-4-v8-epi-{}-vm-100".format(episodes))
    # plt.overall_records_visualization(records['benchmark'], xlabel="Cycle", ylabel="Benchmark Makespan(s)",
    #                                   title="Experiment 3: Benchmark Makespan over Cycle ({} Episodes)".format(
    #                                       episodes))
    #
    # plt.overall_records_visualization(
    #     records['makespan'], xlabel='Cycle', ylabel="Makespan(s)",
    #     title="Experiment 3: Makespan(s) across Cycle ({} Episodes)".format(episodes)
    # )
    plt.plot_episode_stats(stats)
