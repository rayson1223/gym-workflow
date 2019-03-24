from gym_workflow.envs.workflow_wf_sim import WorkflowSimEnv
from gym_workflow.envs.database import DatabaseEnv
from gym.spaces import Discrete, Tuple
import random
import numpy as np
import sys
from gym_workflow.libs.recording import *


class Version9(WorkflowSimEnv):
    """
        @version 7.0:
            - Design based on workflow sim
            - Reduce the state of Q to only 3, {Low, Medium, High}
            - actions scheme same as v7
    """

    def __init__(self):
        # Montage Experiment Variable
        super(Version9, self).__init__()

        self.action_range = 10
        self.cluster_range = 10
        self.action_space = Discrete(self.action_range)
        self.observation_space = Discrete(2)
        self.cluster_size = 1

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
        done = True
        reward = 0.0

        if action == 0:
            self.cluster_size + 1
        elif action == 1:
            self.cluster_size - 1

        # Return all the data collected
        makespan, queue, exec, postscript, cost, vm = self.run_experiment(vm_size=self.action_range,
                                                                             clustering_choice="CSIZE",
                                                                             clustering_method="HORIZONTAL",
                                                                             cluster_size=self.cluster_size)
        overhead = float(queue) + float(postscript)

        self.all_overhead_record.append(overhead)
        self.all_makespan_record.append(float(makespan))
        self.all_exec_record.append(float(exec))

        if not training:
            # Setting up best exec time
            if self.best_overhead is None:
                self.best_overhead = overhead
            if self.last_overhead is None:
                self.last_overhead = overhead

            # Rewarding / Penalty Judgement
            if overhead < np.percentile(self.all_overhead_record, 20):
                self.best_overhead = overhead
                reward = 500
            else:
                reward = -100
            self.last_overhead = overhead

            self.total_reward += reward
            self.last_action = action

        return self._get_obs(), reward, done, {
            "exec": self.all_exec_record,
            "overhead": self.all_overhead_record,
            "makespan": self.all_makespan_record
        }

    def reset(self):
        # 0: Low
        # 1: High
        return random.randint(0, 1)

    def _get_obs(self):
        return random.randint(0, 1)
