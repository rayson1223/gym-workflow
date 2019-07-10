from gym_workflow.envs.montage_wf_env import MontageWfEnv
from gym_workflow.envs.database import DatabaseEnv
from gym.spaces import Discrete, Tuple
import random
import numpy as np
import sys
from gym_workflow.libs.recording import *


class Version9(MontageWfEnv):
    """
        @version 9.0:
            - Update version of V5.0
            - Required update V5 to only take in CS
            - Required verify it's convergence by simulation before deployment
    """

    def __init__(self):
        # Montage Experiment Variable
        super(Version9, self).__init__()

        # 3: cs, 5: cs & cn
        self.action_range = 5
        # 10 = 10 processors available
        self.cluster_range = 10
        self.action_space = Discrete(self.action_range)
        self.observation_space = Discrete(self.cluster_range) #, Discrete(self.cluster_range)
        self.cluster_size = 1
        self.cluster_num = 1

        # Episode Conf
        # Best exec_time: None or 1, depends on reward version
        self.best_overhead = None
        self.last_overhead = None
        self.last_action = None
        self.reward = 0
        self.total_reward = 0.0
        self.is_training = False

        self.exec_records = {}
        self.all_exec_record = list()
        self.all_overhead_record = list()
        self.all_makespan_record = list()
        self.seed()
        self.reset()

    def step(self, action, training=False):
        assert self.action_space.contains(action)
        self.is_training = training
        done = False
        reward = 0.0

        if action == 1:
            self.cluster_size += 1
        elif action == 2:
            self.cluster_size -= 1
        elif action == 3:
            self.cluster_num += 1
        elif action == 4:
            self.cluster_num -= 1

        # Range Guarding Function
        if self.cluster_size <= 0:
            # reward = -100.0
            self.cluster_size = 1
        elif self.cluster_size > 10:
            # reward = -100.0
            self.cluster_size = 10
        elif self.cluster_num <= 0:
            # reward = -100.0
            self.cluster_num = 1
        elif self.cluster_num > 10:
            # reward = -100.0
            self.cluster_num = 10
        else:
            # Return all the data collected
            # status, jb, wt, cwt = self.run_experiment(self.clusters_size, self.clusters_num)

            # Experiment run failed -> High Penalty
            # if not status:
            #     return self._get_obs(), -10, True, {}
            # jb = 0 if jb is None else jb
            # wt = 0 if wt is None else wt
            # cwt = 0 if cwt is None else cwt

            jb = self.run_gen_experiment(self.cluster_size, self.cluster_num)
            # jb = self.run_gen_experiment(self.cluster_size)
            wt = 0
            cwt = 0

            # self.exec_time = jb
            self.exec_time = float(jb)
            overhead = self.exec_time

            self.all_overhead_record.append(jb)
            self.all_makespan_record.append(jb)
            self.all_exec_record.append(jb)

            if not training:
                # Setting up best exec time
                if self.best_overhead is None:
                    self.best_overhead = overhead
                if self.last_overhead is None:
                    self.last_overhead = overhead

                # Rewarding / Penalty Judgement
                if overhead < np.percentile(self.all_overhead_record, 20):
                    self.best_overhead = overhead
                    reward = 200
                else:
                    reward = -100
                self.last_overhead = overhead

            self.total_reward += reward
            self.last_action = action
            if self.total_reward > 2000 or self.total_reward < -1000:
                done = True

        return self._get_obs(), reward, done, {
            "exec": self.exec_time,
            "overhead": self.exec_time,
            "makespan": self.exec_time
        }

    def reset(self):
        if self.last_overhead is not None:
            self.last_overhead = self.exec_time
        self.cluster_size = np.random.randint(1, self.cluster_range)
        self.cluster_num = np.random.randint(1, self.cluster_range)
        self.total_reward = 0
        # For cs only
        # return self.cluster_size
        # For cs & cn
        return self.cluster_size, self.cluster_num

    def _get_obs(self):
        # For cs
        # return self.cluster_size
        # For cs & cn
        return self.cluster_size, self.cluster_num
