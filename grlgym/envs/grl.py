import gym
import py_env
from gym.spaces import Box

class GRLEnv(gym.Env):

    def __init__(self, config_file, test = 0):
        od, o_min, o_max, ad, a_min, a_max = py_env.init(config_file)
        self.observation_space = Box(o_min, o_max)
        self.action_space = Box(a_min, a_max)
        self.test = test
    
    def _seed(self, seed):
        """ Starting the environment """
        if not seed:
            seed = 0
        return py_env.seed(seed)
    
    def _reset(self):
        """ Starting the environment """
        return py_env.start(self.test)

    def _step(self, action):
        """ Next observation """
        obs, reward, terminal, tau = py_env.step(action)
        return obs, reward, terminal, {}
        
    def _close(self):
        py_env.fini()

