import gym
import py_env
from gym.spaces import Box

class GRLEnv(gym.Env):

    def __init__(self, config_file):
        od, o_min, o_max, ad, a_min, a_max = py_env.init(config_file)
        self.observation_space = Box(o_min, o_max)
        self.action_space = Box(a_min, a_max)

    def _reset(self):
        """ Starting the environment """
        return py_env.start(0)

    def _step(self, action):
        """ Next observation """
        obs, reward, terminal, tau = py_env.step(action)
        return obs, reward, terminal, {}
        
    def _close(self):
        py_env.fini()

