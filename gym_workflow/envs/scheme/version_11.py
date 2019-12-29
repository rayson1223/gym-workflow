import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Discrete

from gym_workflow.envs.montage_wf_env import MontageWfEnv


class Version11(MontageWfEnv):
    """
        @version 11.0:

    """

    def __init__(self):
        # Montage Experiment Variable
        super(Version11, self).__init__()

        self.action_range = 30
        self.cluster_range = 30
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
        self.all_benchmark_record = list()
        self.seed()
        self.reset()

    def step(self, action, training=False):
        assert self.action_space.contains(action)
        action += 1
        done = False
        reward = 0.0

        # Return all the data collected
        # makespan = self.run_gen_experiment(action)
        makespan = self.run_cn_gen_experiment(action)
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
            # if self.best_makespan is None:
            #     self.best_makespan = makespan
            # if self.last_makespan is None:
            #     self.last_makespan = makespan

            # Rewarding / Penalty Judgement
            percentile = np.percentile(self.all_makespan_record, 10)
            benchmark = np.mean(self.all_benchmark_record)
            # Calc improve percentage
            if len(self.all_benchmark_record) == 0:
                self.all_benchmark_record.append(percentile)
            elif abs(percentile - benchmark) / benchmark * 100 > 10:
                self.all_benchmark_record.append(percentile)
                benchmark = np.mean(self.all_benchmark_record)
            # else:
                # print(abs(percentile - benchmark) / benchmark * 100)
                # self.all_benchmark_record.append(percentile)
                # benchmark = percentile

            if makespan < benchmark:
                reward = 1
            else:
                reward = -1
            self.last_makespan = makespan

            self.total_reward += reward
            self.last_action = action
            if self.total_reward > 20 or self.total_reward < -20:
                done = True

        return self._get_obs(), reward, done, {
            "exec": makespan,
            "overhead": makespan,
            "makespan": makespan,
            "benchmark": benchmark
        }

    def reset(self):
        self.total_reward = 0.0
        return np.random.randint(1, self.cluster_range + 1)  # , self.clusters_num

    def _get_obs(self):
        return np.random.randint(1, self.cluster_range + 1)
