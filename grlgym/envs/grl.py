import gym
from py_env import PyEnv
from gym.spaces import Box
import numpy as np

class GRLEnv(gym.Env):

    def __init__(self, config_file):
        self.env = PyEnv()
        o_dims, o_min, o_max, ad, a_min, a_max = self.env.init(config_file)
        self.o_dims = o_dims
        self.observation_space = Box(o_min, o_max)
        self.action_space = Box(a_min, a_max)

    def _seed(self, seed):
        """ Starting the environment """
        if not seed:
            seed = 0
        self.env.seed(seed)

    def _reset(self, test=False, **kwargs):
        """ Starting the environment """
        obs = self.env.start(int(test))
        return obs

    def _step(self, action):
        """ Next observation """
        # sometimes provided action is float32, but only float64 is accepted
        obs, reward, terminal, info = self.env.step(action.astype(np.float64))
        return obs, reward, terminal == 2, info

    def _close(self):
        self.env.fini()

    # Own methods
    def report(self, report='test'):
        """ Select in which episodes report (info) is requested """
        idx = ['learn', 'test', 'all'].index(report)
        self.env.report(idx)

    def reconfigure(self, d=None):
        """ Reconfigure the environemnt using the dict """
        self.env.reconfigure(d)

class Leo(GRLEnv):

    def __init__(self, config_file):
        """ Last observation element tells what the forward promotion is """
        self.env = PyEnv()
        o_dims, o_min, o_max, ad, a_min, a_max = self.env.init(config_file)
        self.o_dims = o_dims-1
        self.observation_space = Box(o_min[:-1], o_max[:-1])
        self.action_space = Box(a_min, a_max)

