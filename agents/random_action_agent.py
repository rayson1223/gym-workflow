from gym import wrappers, make
import logging
import numpy as np

logger = logging.getLogger(__name__)
import gym_workflow.envs

if __name__ == '__main__':
	env = make('Montage-v0')

	# You can set the level to logger.DEBUG or logger.WARN if you
	# want to change the amount of output.
	# logger.setLevel(logger.info)

	# You provide the directory to write to (can be an existing
	# directory, including one with existing data -- all monitor files
	# will be namespaced). You can also dump to a tempdir if you'd
	# like: tempfile.mkdtemp().
	outdir = './montage-agent-results'
	# env = wrappers.Monitor(env, directory=outdir, force=True)
	# env.seed()

	sc, fc = 0, 0
	for i_episode in range(100):
		done = False
		env.reset()
		state = env.action_space.sample()
		print("Episode: %s" % i_episode)

		for t in range(100):

			print("\n" * 2)
			env.render()
			action = env.action_space.sample()
			state2, reward, done, info = env.step(action)

			print("Current Reward: %s \t Total Reward: %s" % (reward, info))
			if done:
				if info > 0:
					sc += 1
				else:
					fc += 1
				print("Episode finished after {} timesteps".format(t + 1))
				print("\n" * 5)
				break
		print("Success Count: %s \t Failed Count: %s" % (sc, fc))
