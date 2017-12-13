import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
import py_env

class GRLEnv(gym.Env):

    def __init__(self, config_file):
        py_env.init(config_file)

    def __del__(self):
        py_env.fini()
        
    def _seed(self, seed=None):
        """ Sets the seed for this env's random number generator(s). """

    def _step(self, action):
        obs, reward, terminal, tau = py_env.step(action)
        return obs, reward, terminal, {}

    def _reset(self):
        """ Restarting the environment """
        return py_env.start(0)

    def _render(self, mode='human', close=False):
        """ No viewer supported at the moment. """

