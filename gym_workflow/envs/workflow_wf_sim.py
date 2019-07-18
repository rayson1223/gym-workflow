import gym
from gym.utils import seeding
import time
import subprocess
import os


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
    def run_experiment(vm_size=20, clustering_choice="NONE", clustering_method="NONE", cluster_size=0):
        cmd = "java -jar {} {} {} {} {} {}". \
            format(
            os.getcwd() + "/../gym_workflow/libs/workflowsim/WorkflowSim-1.0.jar",
            vm_size, clustering_choice, clustering_method, cluster_size,
            os.getcwd() + "/../gym_workflow/libs/workflowsim/dax/Montage_1000.xml"
        )
        output = subprocess.getoutput(cmd).strip().split('\n')
        return output[len(output) - 1].split()
