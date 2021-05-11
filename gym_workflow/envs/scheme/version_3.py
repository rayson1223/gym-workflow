from gym_workflow.envs.montage_wf_env import MontageWfEnv
from gym_workflow.envs.database import DatabaseEnv
from gym.spaces import Discrete, Tuple
import numpy as np
import random


class Version3(MontageWfEnv):
    """
        @version 3.0:
            Prove this fail! -> up, down, maintain action:
    """

    def __init__(self, degree=0.1, band_num=1, db_dir=".pegasus/workflow.db"):
        # Montage Experiment Variable
        super(Version3, self).__init__()
        self.degree = degree
        self.is_clusters_size = False
        self.is_clusters_num = True
        self.band_num = band_num

        # Setting database connection
        # self.db = DatabaseEnv(db_dir)

        self.action_space = Discrete(3)

        self.observation_space = Discrete(3)  # , Discrete(8), Discrete(3)

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

        reward = 0.0
        self.last_action = action
        done = False
        if action == 1:
            self.clusters_size += 1
        elif action == 2:
            self.clusters_size -= 1

        # Range Guarding Function
        if self.clusters_size <= 0:
            reward -= 1.0
            self.clusters_size = 1
        elif self.clusters_size > 30:
            reward -= 1.0
            self.clusters_size = 30
        else:
            # res = self.run_static_experiment(self.clusters_size, self.clusters_num)
            res = self.run_demo_cn_gen_experiment(self.clusters_size)

            self.all_makespan_record.append(res)
            self.exec_time = res

            # Rewarding / Penalty Judgement
            percentile = np.percentile(self.all_makespan_record, 10)

            if self.exec_time < percentile:
                reward = 2
            else:
                reward = -1
            self.total_reward += reward
            if self.total_reward > 10:
                done = True
        return self._get_obs(), reward, done, {
            "exec": self.exec_time,
            "overhead": self.exec_time,
            "makespan": self.exec_time,
            "benchmark": np.percentile(self.all_makespan_record, 10)
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
        return self.clusters_size

    def _get_obs(self):
        return self.clusters_size
