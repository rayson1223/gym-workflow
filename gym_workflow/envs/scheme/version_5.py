from gym_workflow.envs.montage_wf_env import MontageWfEnv
from gym_workflow.envs.database import DatabaseEnv
from gym.spaces import Discrete
import sys
from gym_workflow.lib.recording import *
import numpy as np


class Version5(MontageWfEnv):
    """
        @version 5.0:
            - Run with real env
            - No Sampling
    """

    def __init__(self):
        # Montage Experiment Variable
        super(Version5, self).__init__()
        self.action_range = 20
        self.cluster_range = 20
        self.action_space = Discrete(self.action_range)
        self.observation_space = Discrete(self.cluster_range)

        # Episode Conf
        # Best exec_time: None or 1, depends on reward version
        self.best_exec_time = None
        self.last_exec_time = None
        self.last_action = None
        self.last_reward = None
        self.total_reward = 0.0
        self.all_makespan_record = list()
        self.all_benchmark_record = list()
        # 0: Ntg, 1: improve, 2: degrade
        self.is_improve = 0
        self.seed()
        self.reset()

    def step(self, action, training=False):
        assert self.action_space.contains(action)

        action += 1
        done = False
        reward = 0.0

        # res = self.run_static_experiment(self.clusters_size, self.clusters_num)
        res = self.run_demo_cn_gen_experiment(action)

        self.all_makespan_record.append(res)
        self.exec_time = res

        # Rewarding / Penalty Judgement
        percentile = np.percentile(self.all_makespan_record, 10)
        benchmark = np.mean(self.all_benchmark_record)
        # Calc improve percentage
        if len(self.all_benchmark_record) == 0:
            self.all_benchmark_record.append(percentile)
        elif abs(percentile - benchmark) / benchmark * 100 > 10:
            self.all_benchmark_record.append(percentile)
            benchmark = np.mean(self.all_benchmark_record)
        # print(benchmark)
        # if self.exec_time < np.percentile(self.all_makespan_record, 10):
        if self.exec_time < benchmark:
            reward = 10
        else:
            reward = -1
        self.total_reward += reward
        if self.total_reward > 200:
            done = True
        return self._get_obs(), reward, done, {
            "exec": self.exec_time,
            "overhead": self.exec_time,
            "makespan": self.exec_time,
            "benchmark": benchmark
        }

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        init_msg = "Current Experiment Parameters: degree: %d\t cluster.size: %d\t cluster.num: %d\t\n" % (
            self.degree, self.clusters_size, self.clusters_num)
        outfile.write(init_msg)
        # if self.last_action is not None:
        # 	cs, cn = action
        result_str = "Current Execution Time: \t"
        expect_str = "Best Execution Time: \t"
        action_str = "Current Action: \t"
        # Process Outputs
        outfile.write(result_str + (" %s " % self.exec_time) + "\n")
        outfile.write(expect_str + (" %s " % self.best_exec_time) + "\n")
        outfile.write(action_str + (" %s " % self.last_action) + "\n")

        return outfile

    def reset(self):
        # Reset method should always return a new sets of episode settings
        # self.degree = 0.1
        # self.clusters_size = 1
        # self.clusters_num = 1
        # self.band_num = 1
        # self.best_exec_time = None
        if self.exec_time is not None:
            self.last_exec_time = self.exec_time
        self.wall_time = None
        self.cum_wall_time = None
        self.total_reward = 0
        self.clusters_size = np.random.randint(1, 30)
        self.clusters_num = np.random.randint(1, 30)

        # print("Environment had been reset!")
        return np.random.randint(1, self.cluster_range + 1)  # , self.clusters_num

    def _get_obs(self):
        return np.random.randint(1, self.cluster_range + 1)
