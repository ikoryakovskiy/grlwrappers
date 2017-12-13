import gym
from py_env import PyEnv
from gym.spaces import Box

class GRLEnv(gym.Env):

    def __init__(self, config_file, test = 0):
        self.env = PyEnv()
        od, o_min, o_max, ad, a_min, a_max = self.env.init(config_file)
        self.observation_space = Box(o_min, o_max)
        self.action_space = Box(a_min, a_max)
        self.test = test

    def _seed(self, seed):
        """ Starting the environment """
        if not seed:
            seed = 0
        return self.env.seed(seed)
    
    def _reset(self):
        """ Starting the environment """
        return self.env.start(self.test)

    def _step(self, action):
        """ Next observation """
        obs, reward, terminal, tau = self.env.step(action)
        return obs, reward, terminal, {}
        
    def _close(self):
        self.env.fini()
