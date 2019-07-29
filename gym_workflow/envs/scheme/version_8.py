from gym_workflow.envs.workflow_wf_sim import WorkflowSimEnv
from gym_workflow.envs.database import DatabaseEnv
from gym.spaces import Discrete, Tuple
import random
import numpy as np
import sys
from gym_workflow.libs.recording import *


class Version8(WorkflowSimEnv):
    """
        @version 8.0:
            - Design based on workflow sim
            - Reward scheme same as v7
            - Focus on system overhead
    """

    def __init__(self):
        # Montage Experiment Variable
        super(Version8, self).__init__()

        self.action_range = 100
        self.cluster_range = 100
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
        makespan, queue, exec, postscript, cluster, wen = self.run_experiment(vm_size=self.action_range,
                                                                          clustering_choice="CSIZE",
                                                                             clustering_method="HORIZONTAL",
                                                                             cluster_size=action)
        overhead = float(queue) + float(postscript) + float(wen) + float(cluster)
        makespan = float(makespan)
        self.all_overhead_record.append(overhead)
        self.all_makespan_record.append(float(makespan))
        self.all_exec_record.append(float(exec))

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
            else:
                reward = -5
            self.last_makespan = makespan

            self.total_reward += reward
            self.last_action = action
            if self.total_reward > 50 or self.total_reward < -100:
                done = True

        return self._get_obs(), reward, done, {
            "exec": exec,
            "overhead": overhead,
            "makespan": makespan,
            "queue": queue,
            "postscript": postscript,
            "cluster": cluster,
            "wen": wen,
            "benchmark": np.percentile(self.all_makespan_record, 20)
        }

    def reset(self):
        self.total_reward = 0.0
        # for x in self.all_makespan_record:
        #     if x > np.percentile(self.all_makespan_record, 20):
        #         self.all_makespan_record.remove(x)
        return np.random.randint(1, (self.cluster_range+1))  # , self.clusters_num

    def _get_obs(self):
        return np.random.randint(1, (self.cluster_range+1))
