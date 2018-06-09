import gym
from py_env import PyEnv
from gym.spaces import Box
import numpy as np
import collections
import random

class GRLEnv(gym.Env):

    def __init__(self, config_file):
        self.env = PyEnv()
        o_dims, o_min, o_max, a_dims, a_min, a_max = self.env.init(config_file)
        self.o_dims = o_dims
        self.a_dims = a_dims
        self.observation_space = Box(o_min, o_max)
        self.action_space = Box(a_min, a_max)
        self.measurment_noise = None
        self.actuation_noise = None

    def _seed(self, seed):
        """ Starting the environment """
        if not seed:
            seed = 0
        self.env.seed(seed)

    def _reset(self, test=False, **kwargs):
        """ Starting the environment """
        obs = self.env.start(int(test))
        obs = self._measurment_noise(obs)
        return obs

    def _step(self, action):
        """ Next observation """
        action = self._actuation_noise(action)
        # sometimes provided action is float32, but only float64 is accepted
        obs, reward, terminal, info = self.env.step(action.astype(np.float64))
        obs = self._measurment_noise(obs)
        return obs, reward, terminal, info

    def _close(self):
        self.env.fini()

    # Own methods
    def report(self, report='test'):
        """ Select in which episodes report (info) is requested """
        idx = ['learn', 'test', 'all'].index(report)
        self.env.report(idx)

    def reconfigure(self, d=None):
        """ Reconfigure the environemnt using the dict """
        if d:
            if 'action' in d.keys():
                self.env.reconfigure(d)
            if 'measurment_noise' in d.keys():
                self.measurment_noise = d['measurment_noise']
                print('Setting measurment noise to ' + str(self.measurment_noise))
            if 'actuation_noise' in d.keys():
                self.actuation_noise = d['actuation_noise']
                print('Setting actuation noise to ' + str(self.actuation_noise))

    def _measurment_noise(self, obs):
        if self.measurment_noise:
            if isinstance(self.measurment_noise, (collections.Sequence, np.ndarray)):
                assert(len(self.measurment_noise) == self.o_dims)
                for i in range(self.o_dims):
                    obs[i] += random.uniform(-self.measurment_noise[i], self.measurment_noise[i])
                    obs[i] = max(min(obs[i], self.observation_space.high[i]), self.observation_space.low[i])
            else:
                for i in range(self.o_dims):
                    obs[i] += random.uniform(-self.measurment_noise, self.measurment_noise)
                    obs[i] = max(min(obs[i], self.observation_space.high[i]), self.observation_space.low[i])
        return obs


    def _actuation_noise(self, action):
        if self.actuation_noise:
            if isinstance(self.actuation_noise, (collections.Sequence, np.ndarray)):
                assert(len(self.actuation_noise) == self.a_dims)
                for i in range(self.a_dims):
                    action[i] += random.uniform(-self.actuation_noise[i], self.actuation_noise[i])
                    action[i] = max(min(action[i], self.action_space.high[i]), self.action_space.low[i])
            else:
                for i in range(self.a_dims):
                    action[i] += random.uniform(-self.actuation_noise, self.actuation_noise)
                    action[i] = max(min(action[i], self.action_space.high[i]), self.action_space.low[i])
        return action


class Leo(GRLEnv):

    def __init__(self, config_file):
        """ Last observation element tells what the forward promotion is """
        super().__init__(config_file)
        self.o_dims = self.o_dims-1
        self.observation_space = Box(self.observation_space.low[:-1], self.observation_space.high[:-1])

