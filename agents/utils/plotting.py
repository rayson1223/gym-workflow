import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.transforms as transforms

matplotlib.style.use('ggplot')
import itertools
from mpl_toolkits.mplot3d import Axes3D


def overall_records_visualization(records_list, xlabel="", ylabel="", title="", show_plot=True):
    plt.clf()
    plt.figure(figsize=(50, 5))
    plt.plot(records_list)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig('./plots/{}.png'.format(title))
    if show_plot:
        plt.show()


def episode_records_boxplot_visualization(records_list, xlabel="", ylabel="", title="", show_plot=True, showfliers=False):
    plt.clf()
    fig = plt.figure(figsize=(50, 5))
    ax = fig.add_subplot(111)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    ax.boxplot(x.values(), showfliers=showfliers)

    plt.savefig("./plots/{}.png".format(title))
    if show_plot:
        plt.show()


def plot_q_learning_value_function(Q, xlabel="", ylabel="", title="Q Value Function", show_plot=True, save_plot=False):
    """
    Plots the value function as a surface plot.
    """
    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    # Find value for all (x, y) coordinates
    Z = np.apply_along_axis(lambda _: V[(_[0], _[1])], 2, np.dstack([X, Y]))

    def plot_surface(X, Y, Z, title):
        def find_min_max_range(values):
            a = values.reshape(values.size)
            return min(a), max(a)

        minV, maxV = find_min_max_range(Z)
        fig = plt.figure(figsize=(5, 3))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, vmin=minV, vmax=maxV)
        ax.set_xlabel('Cluster Size')
        ax.set_ylabel('Cluster Number')
        ax.set_zlabel('Value')
        ax.set_title(title)
        ax.view_init(ax.elev, -120)
        fig.colorbar(surf)
        fig.savefig(title)
        if show_plot:
            plt.show()

    plot_surface(X, Y, Z, title)


def v1_plot_action_value(Q, title="", show_plot=True):
    # min_y = min(min(v) for v in Q.values())
    # max_y = max(max(v) for v in Q.values())

    # y_range = np.arange(min_y, max_y + 1)
    fig = plt.figure(figsize=(15, 5))
    keys = []
    maintain_value = []
    add_cs_value = []
    add_cn_value = []
    minus_cs_value = []
    minus_cn_value = []

    if 'key' in Q:
        del Q['key']
    for k, v in Q.items():
        if not callable(v):
            # keys.append(', '.join(map(str, k)))
            keys.append(k)
            maintain_value.append(v[0])
            add_cs_value.append(v[1])
            minus_cs_value.append(v[2])
            # add_cn_value.append(v[3])
            # minus_cn_value.append(v[4])
    plt.clf()
    plt.plot(keys, maintain_value, '-', label="Maintain")
    plt.plot(keys, add_cs_value, '^--', label="+CN")
    plt.plot(keys, minus_cs_value, 'v--', label="-CN")
    # plt.plot(keys, add_cn_value, '^:', label="+CN")
    # plt.plot(keys, minus_cn_value, 'v:', label="-CN")
    # plt.xticks(keys, rotation='vertical')

    plt.xlabel('Cluster Number')
    plt.ylabel('Q Values')

    plt.title(title)
    plt.legend()
    fig.savefig('plots/{}.png'.format(title))
    if show_plot:
        plt.show()


def v1_plot_episode_stats(stats, smoothing_window=10, noshow=False):
    # Plot the episode length over time
    epiMean = np.average(stats.episode_lengths)
    # fig1 = plt.figure(figsize=(10, 5))
    fig1, axes = plt.subplots()
    axes.plot(stats.episode_lengths)
    axes.axhline(y=epiMean, color='black', ls='--')
    trans = transforms.blended_transform_factory(axes.get_yticklabels()[0].get_transform(), axes.transData)
    axes.text(0, epiMean, "{:.0f}".format(epiMean), color="black", transform=trans, ha="right", va="center")
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    if noshow:
        fig1.savefig("Experiment 1: Episode Length over Time")
        plt.close(fig1)
    else:
        fig1.savefig("Experiment 1: Episode Length over Time")
        fig1.show()

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10, 5))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.axhline(y=10, color='black', ls='--')
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    if noshow:
        fig2.savefig("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
        plt.close(fig2)
    else:
        fig2.savefig("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
        fig2.show()

    # Plot time steps and episode number
    fig3 = plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(stats.episode_lengths), np.arange(len(stats.episode_lengths)))
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    if noshow:
        fig3.savefig("Experiment 1: Episode per time step")
        plt.close(fig3)
    else:
        fig3.savefig("Experiment 1: Episode per time step")
        fig3.show()

    fig4 = plt.figure(figsize=(10, 5))
    rewards_smoothed = pd.Series(stats.episode_total_reward).rolling(smoothing_window,
                                                                     min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Total Reward (Smoothed)")
    plt.title("Episode Total Reward over Time (Smoothed over window size {})".format(smoothing_window))
    if noshow:
        fig4.savefig("Experiment 1: Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
        plt.close(fig4)
    else:
        fig4.savefig("Experiment 1: Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
        fig4.show()

    return fig1, fig2, fig3


def plot_exp_2_action_value(Q, title="", show_plot=True):
    if 'key' in Q.keys():
        del Q['key']
    min_x = min(k for k in Q.keys())
    max_x = max(k for k in Q.keys())
    # tmp = []
    # for y in V.values():
    #     tmp += y
    # min_y = min(tmp)
    # max_y = max(tmp)

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_x, max_x + 1)
    X, Y = np.meshgrid(x_range, y_range)

    # Find value for all (x, y) coordinates
    # Z = np.apply_along_axis(lambda _: V[_[0]][_[1]-1], 2, np.dstack([X, Y]))
    Z = np.zeros(shape=(max_x, max_x))
    for x, v in Q.items():
        for i, y in enumerate(v, start=0):
            Z[x - 1, i] = y

    def plot_surface(X, Y, Z, title):
        def find_min_max_range(values):
            a = values.reshape(values.size)
            return min(a), max(a)

        minV, maxV = find_min_max_range(Z)
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, vmin=minV, vmax=maxV)
        ax.set_xlabel('Available action (cluster number)')
        ax.set_ylabel('State (cluster number)')
        ax.set_zlabel('Q-Value (preference action)')
        ax.set_title(title)
        ax.view_init(ax.elev, -120)
        fig.colorbar(surf)
        fig.savefig("./plots/{}.png".format(title))
        # if show_plot:
        plt.show()

    plot_surface(X, Y, Z, title)


def plot_exp_3_action_value(Q, title="", show_plot=True):
    min_x = min(k for k in Q.keys())
    max_x = max(k for k in Q.keys())
    # tmp = []
    # for y in V.values():
    #     tmp += y
    # min_y = min(tmp)
    # max_y = max(tmp)

    # Original
    x_range = np.arange(0, 99 + 1)
    y_range = np.arange(0, 99 + 1)

    X, Y = np.meshgrid(x_range, y_range)

    # Find value for all (x, y) coordinates
    # Z = np.apply_along_axis(lambda _: V[_[0]][_[1]-1], 2, np.dstack([X, Y]))
    Z = np.zeros(shape=(100, 100))
    for x, v in Q.items():
        for i, y in enumerate(v, start=0):
            Z[x - 1, i] = y

    def plot_surface(X, Y, Z, title):
        def find_min_max_range(values):
            a = values.reshape(values.size)
            return min(a), max(a)

        minV, maxV = find_min_max_range(Z)
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, vmin=minV, vmax=maxV)
        ax.set_xlabel('Available action (cluster number)')
        ax.set_ylabel('State (cluster number)')
        ax.set_zlabel('Q-Value (preference action)')
        ax.set_title(title)
        ax.view_init(ax.elev, -120)
        fig.colorbar(surf)
        fig.savefig("./plots/{}.png".format(title))
        # if show_plot:
        plt.show()

    plot_surface(X, Y, Z, title)


def plot_simple_line(records, xlabel="", ylabel="", title="", show_plot=True):
    plt.plot(pd.Series(records).rolling(5, min_periods=5).mean())
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if show_plot:
        plt.show()
    plt.savefig('plots/{}'.format(title))


def plot_boxplot(values=[], labels=[], xlabel="", ylabel="", title="", outliers=False, show_plot=True):
    fig = plt.figure(figsize=(20, 15))
    ax = fig.add_subplot(111)
    ax.boxplot(values, labels=labels, showfliers=outliers)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if show_plot:
        plt.show()
    plt.savefig('plots/{}'.format(title))


def plot_value_function(V, title="Value Function", show_plot=True):
    """
    Plots the value function as a surface plot.
    """
    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    # Find value for all (x, y) coordinates
    Z = np.apply_along_axis(lambda _: V[(_[0], _[1])], 2, np.dstack([X, Y]))

    def plot_surface(X, Y, Z, title):
        def find_min_max_range(values):
            a = values.reshape(values.size)
            return min(a), max(a)

        minV, maxV = find_min_max_range(Z)
        fig = plt.figure(figsize=(5, 3))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, vmin=minV, vmax=maxV)
        ax.set_xlabel('Cluster Size')
        ax.set_ylabel('Cluster Number')
        ax.set_zlabel('Value')
        ax.set_title(title)
        ax.view_init(ax.elev, -120)
        fig.colorbar(surf)
        fig.savefig(title)
        if show_plot:
            plt.show()

    plot_surface(X, Y, Z, title)


def plot_line_value(Q, title="default", show_plot=True):
    if 'key' in Q.keys():
        del Q['key']
    min_x = min(int(k) for k in Q.keys())
    max_x = max(int(k) for k in Q.keys())
    min_y = min(min(v) for v in Q.values())
    max_y = max(max(v) for v in Q.values())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    fig = plt.figure(figsize=(8, 5))
    plt.plot(Q.keys(), [v[0] for v in Q.values()], '-', label="Maintain")
    plt.plot(Q.keys(), [v[1] for v in Q.values()], '^--', label="+CS")
    plt.plot(Q.keys(), [v[2] for v in Q.values()], 'v--', label="-CS")
    # plt.plot(Q.keys(), [v[0] for v in Q.values()], 'b-', label="Remain")
    # plt.plot(Q.keys(), [v[1] for v in Q.values()], 'g--', label="Minus")
    # plt.plot(Q.keys(), [v[1] for v in Q.values()], 'r-.', label="Remain")

    plt.xlabel('Cluster Size')
    plt.ylabel('Action Values')

    plt.title(title)
    plt.legend()
    fig.savefig('plots/{}'.format(title))
    if show_plot:
        plt.show()


def plot_episode_stats(stats, smoothing_window=10, noshow=False):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10, 5))
    plt.plot(stats.episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    if noshow:
        fig1.savefig("Q Learning: Episode Length over Time")
        plt.close(fig1)
    else:
        fig1.savefig("Q Learning: Episode Length over Time")
        fig1.show()

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10, 5))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    if noshow:
        fig2.savefig("Q Learning: Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
        plt.close(fig2)
    else:
        fig2.savefig("Q Learning: Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
        fig2.show()

    # Plot time steps and episode number
    fig3 = plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(stats.episode_lengths), np.arange(len(stats.episode_lengths)))
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    if noshow:
        fig3.savefig("Q Learning: Episode per time step")
        plt.close(fig3)
    else:
        fig3.savefig("Q Learning: Episode per time step")
        fig3.show()

    fig4 = plt.figure(figsize=(10, 5))
    rewards_smoothed = pd.Series(stats.episode_total_reward).rolling(smoothing_window,
                                                                     min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Total Reward (Smoothed)")
    plt.title("Episode Total Reward over Time (Smoothed over window size {})".format(smoothing_window))
    if noshow:
        fig4.savefig("Q Learning: Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
        plt.close(fig4)
    else:
        fig4.savefig("Q Learning: Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
        fig4.show()

    return fig1, fig2, fig3
