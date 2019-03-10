import gym
from gym.utils import seeding
import time
import subprocess


class WorkflowSimEnv(gym.Env):
    # General Workflow Environment
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        self.queue_time = None
        self.exec_time = None
        self.postscripted_time = None
        self.makespan_time = None
        self.np_random = None
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human'):
        pass

    @staticmethod
    def run_cs_experiment(vm_size=20, clustering_method="NONE", cluster_size=1):
        cmd = "java -jar /Users/rayson/Documents/master/gym-workflow/gym_workflow/libs/workflowsim/WorkflowSim-cs.jar {} {} {}". \
            format(vm_size, clustering_method, cluster_size)
        output = subprocess.getoutput(cmd).strip().split('\n')
        return output[len(output) - 1].split()

    @staticmethod
    def run_cn_experiment(vm_size=20, clustering_method="NONE", cluster_size=1):
        cmd = "java -jar /Users/rayson/Documents/master/gym-workflow/gym_workflow/libs/workflowsim/WorkflowSim-cn.jar {} {} {}". \
            format(vm_size, clustering_method, cluster_size)
        output = subprocess.getoutput(cmd).strip().split('\n')
        return output[len(output) - 1].split()
