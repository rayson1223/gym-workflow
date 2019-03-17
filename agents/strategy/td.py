import numpy as np
import pandas as pd
import json
import itertools
import sys
from collections import defaultdict, OrderedDict
from collections import namedtuple
from agents.policy.montage_workflow_policy_factory import MontageWorkflowPolicyFactory
from gym_workflow.libs.recording import *
import agents.utils.plotting as plt


class TD:

    @staticmethod
    def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1, training_episode=0, log_file="training-records.csv"):
        """
        Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
        while following an epsilon-greedy policy

        Args:
                env: OpenAI environment.
                num_episodes: Number of episodes to run for.
                discount_factor: Gamma discount factor.
                alpha: TD learning rate.
                epsilon: Chance the sample a random action. Float betwen 0 and 1.
                training_episode: Number of episode to let the environment training
                log_file: Name of the log file

        Returns:
                A tuple (Q, episode_lengths).
                Q is the optimal action-value function, a dictionary mapping state -> action values.
                stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
        """
        EpisodeStats = namedtuple("Stats",
                                  ["episode_lengths", "episode_rewards", "episode_total_reward", "episode_action"])
        # The final action-value function.
        # A nested dictionary that maps state -> (action -> action-value).
        Q = defaultdict(lambda: np.zeros(env.action_space.n))

        # Keeps track of useful statistics
        stats = EpisodeStats(
            episode_lengths=np.zeros(num_episodes),
            episode_rewards=np.zeros(num_episodes),
            episode_total_reward=np.zeros(num_episodes),
            episode_action=np.zeros(num_episodes)
        )

        # if continuous get -negative termination, let's put an end
        termination_count = 0

        # The policy we're following
        policy = MontageWorkflowPolicyFactory().make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

        if training_episode > 0:
            for epi in range(training_episode):
                epi_record = {
                    "exec": [],
                    "overhead": [],
                    "makespan": []
                }
                print("Training in progress: {}".format(epi_record))
                state = env.reset()
                for t in itertools.count():
                    action_probs = policy(state)
                    action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                    next_state, reward, done, records = env.step(action, training=True)
                    epi_record['exec'].append(records['exec'])
                    epi_record['overhead'].append(records['overhead'])
                    epi_record['makespan'].append(records['makespan'])
                    if done or t + 1 > 100:
                        break
                write_record(
                    [epi, json.dumps(epi_record)], header=['episode', 'records'],
                    filename="training_exec_records.csv"
                )

        for i_episode in range(num_episodes):
            # Print out which episode we're on, useful for debugging.
            if (i_episode + 1) % 10 == 0:
                print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
                plt.plot_simple_line(exec_records["overhead"], xlabel="Episode", ylabel="Overhead(s)",
                                     title="Episode vs Overhead(s) - {} Episode".format(i_episode+1))
                plt.plot_episode_stats(stats)
                sys.stdout.flush()

            # Reset the environment and pick the first action
            state = env.reset()

            pd.Series(stats.episode_rewards).to_csv(path="records/episode_reward.csv")
            pd.Series(stats.episode_lengths).to_csv(path="records/episode_lengths.csv")
            pd.Series(stats.episode_total_reward).to_csv(path="records/episode_total_reward.csv")

            # Reset episode records
            exec_records = {
                "exec": [],
                "overhead": [],
                "makespan": []
            }

            for t in itertools.count():
                # Take a step
                action_probs = policy(state)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                next_state, reward, done, records = env.step(action, training=False)
                exec_records['exec'].append(records['exec'])
                exec_records['overhead'].append(records['overhead'])
                exec_records['makespan'].append(records['makespan'])

                # Update statistics
                stats.episode_rewards[i_episode] += reward
                stats.episode_lengths[i_episode] = t
                stats.episode_total_reward[i_episode] = sum(stats.episode_rewards[0:i_episode + 1])
                stats.episode_action[i_episode] = action

                # TD Update
                best_next_action = np.argmax(Q[next_state])
                td_target = reward + discount_factor * Q[next_state][best_next_action]
                td_delta = td_target - Q[state][action]
                Q[state][action] += alpha * td_delta
                write_record(
                    [i_episode, state, action, next_state, Q, json.dumps(action_probs.tolist()), reward],
                    header=['episode', 'state', 'action', 'next_state', 'Q value', 'action prob', 'reward'],
                    filename=log_file
                )
                if done or t + 1 > 100:
                    if reward < 0:
                        termination_count += 1
                    else:
                        termination_count = 0
                    break
                state = next_state
            # Write down the episode records at the end of episode
            write_record(
                [i_episode, json.dumps(exec_records)], header=['episode', 'records'],
                filename="execution_records.csv"
            )
            if termination_count >= 10:
                break
        return Q, stats, records

    @staticmethod
    def sarsa(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
        """
        SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.

        Args:
                env: OpenAI environment.
                num_episodes: Number of episodes to run for.
                discount_factor: Gamma discount factor.
                alpha: TD learning rate.
                epsilon: Chance the sample a random action. Float betwen 0 and 1.

        Returns:
                A tuple (Q, stats).
                Q is the optimal action-value function, a dictionary mapping state -> action values.
                stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
        """

        # The final action-value function.
        # A nested dictionary that maps state -> (action -> action-value).
        Q = defaultdict(lambda: np.zeros(env.action_space.n))
        EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])

        # Keeps track of useful statistics
        stats = EpisodeStats(
            episode_lengths=np.zeros(num_episodes),
            episode_rewards=np.zeros(num_episodes))

        # The policy we're following
        policy = MontageWorkflowPolicyFactory().make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

        for i_episode in range(num_episodes):
            # Print out which episode we're on, useful for debugging.
            if (i_episode + 1) % 100 == 0:
                print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
                sys.stdout.flush()

            # Reset the environment and pick the first action
            state = env.reset()
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

            # One step in the environment
            for t in itertools.count():
                # Take a step
                next_state, reward, done, _ = env.step(action)

                # Pick the next action
                next_action_probs = policy(next_state)
                next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)

                # Update statistics
                stats.episode_rewards[i_episode] += reward
                stats.episode_lengths[i_episode] = t

                # TD Update
                td_target = reward + discount_factor * Q[next_state][next_action]
                td_delta = td_target - Q[state][action]
                Q[state][action] += alpha * td_delta

                if done:
                    break

                action = next_action
                state = next_state

        return Q, stats
