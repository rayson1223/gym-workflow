from gym_workflow.envs.montage_wf_env import MontageWfEnv
from gym_workflow.envs.database import DatabaseEnv
from gym.spaces import Discrete, Tuple
import random
import numpy as np
import sys
from gym_workflow.libs.recording import *


class Version10(MontageWfEnv):
    """
        @version 10.0:
            - Design based on pegasus
            - Reward scheme same as v8
            - Action scheu
    """

    def __init__(self):
        # Montage Experiment Variable
        super(Version10, self).__init__()

        self.action_range = 10
        self.cluster_range = 10
        self.action_space = Discrete(self.action_range)
        self.observation_space = Discrete(self.cluster_range)
        self.cluster_size = 1

        # Episode Conf
        # Best exec_time: None or 1, depends on reward version
        self.best_makespan = None
        self.last_makespan = None
        self.last_action = None
        self.reward = 0
        self.total_reward = 0.0

        self.exec_records = {}
        self.all_exec_record = list()
        self.all_overhead_record = list()
        self.all_makespan_record = list()
        self.seed()
        self.reset()

    def step(self, action, training=False):
        assert self.action_space.contains(action)
        action += 1
        done = False
        reward = 0.0

        # Return all the data collected
        makespan = self.run_gen_experiment(action)
        # status, jb, wt, cwt = self.run_experiment(action)

        # Experiment run failed -> High Penalty
        # if not status:
        #     return self._get_obs(), 0, True, {}
        #
        # jb = 0 if jb is None else jb
        # wt = 0 if wt is None else wt
        # cwt = 0 if cwt is None else cwt
        #
        # makespan = jb
        self.all_makespan_record.append(makespan)

        if not training:
            # Setting up best exec
            if self.best_makespan is None:
                self.best_makespan = makespan
            if self.last_makespan is None:
                self.last_makespan = makespan

            # Rewarding / Penalty Judgement
            if makespan < np.percentile(self.all_makespan_record, 20):
                self.best_makespan = makespan
                reward = 10
            # elif makespan < np.percentile(self.all_makespan_record, 40):
            #     reward = 1
            else:
                reward = -5
            self.last_makespan = makespan

            self.total_reward += reward
            self.last_action = action
            if self.total_reward > 50 or self.total_reward < -100:
                done = True

        return self._get_obs(), reward, done, {
            "exec": makespan,
            "overhead": makespan,
            "makespan": makespan,
            "benchmark": np.percentile(self.all_makespan_record, 20)
        }

    def reset(self):
        self.total_reward = 0.0
        # # This part is crucial, as it determine what data going to be stay and what's going to be remove to
        # # maintain the data stability
        # self.all_makespan_record.sort()
        # for x in self.all_makespan_record:
        #     # if wanna get reward while mantain the data, must determine a suitable range of data
        #     # if x < np.percentile(self.all_makespan_record, 20) or x > np.percentile(self.all_makespan_record, 80):
        #     # Unable to obtain proper reward if without remove the extra datas
        #     if x > np.percentile(self.all_makespan_record, 40):
        #         self.all_makespan_record.remove(x)
        # if len(self.all_makespan_record) > 20:
        #     while len(self.all_makespan_record) > 20:
        #         # try to remove the median data
        #         # self.all_makespan_record.remove(self.all_makespan_record[int(len(self.all_makespan_record) / 2) + 1])
        #         # self.all_makespan_record.remove(self.all_makespan_record[np.random.randint(1, len(self.all_makespan_record))])
        #         self.all_makespan_record.remove(self.all_makespan_record[0])
        #         # self.all_makespan_record.remove(self.all_makespan_record[len(self.all_makespan_record)-1])
        return np.random.randint(1, self.cluster_range + 1)  # , self.clusters_num

    def _get_obs(self):
        return np.random.randint(1, self.cluster_range + 1)
