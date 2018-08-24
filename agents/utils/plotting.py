import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

matplotlib.style.use('ggplot')
import itertools
from mpl_toolkits.mplot3d import Axes3D


def plot_value_function(V, title="Value Function"):
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
		fig = plt.figure(figsize=(20, 10))
		ax = fig.add_subplot(111, projection='3d')
		surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, vmin=minV, vmax=maxV)
		ax.set_xlabel('Cluster Size')
		ax.set_ylabel('Cluster Number')
		ax.set_zlabel('Value')
		ax.set_title(title)
		ax.view_init(ax.elev, -120)
		fig.colorbar(surf)
		fig.savefig(title)
		plt.show()

	plot_surface(X, Y, Z, title)


def plot_episode_stats(stats, smoothing_window=10, noshow=False):
	# Plot the episode length over time
	fig1 = plt.figure(figsize=(10, 5))
	plt.plot(stats.episode_lengths)
	plt.xlabel("Episode")
	plt.ylabel("Episode Length")
	plt.title("Episode Length over Time")
	if noshow:
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
		plt.close(fig3)
	else:
		fig3.savefig("Q Learning: Episode per time step")
		fig3.show()

	return fig1, fig2, fig3
