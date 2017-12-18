import gym
from py_env import PyEnv
from gym.spaces import Box

class GRLEnv(gym.Env):

    def __init__(self, config_file):
        self.env = PyEnv()
        #self.spec.env_id = config_file
        od, o_min, o_max, ad, a_min, a_max = self.env.init(config_file)
                        
        self.observation_space = Box(o_min, o_max)
        self.action_space = Box(a_min, a_max)
        self.test = False

    def _seed(self, seed):
        """ Starting the environment """
        if not seed:
            seed = 0
        self.env.seed(seed)
    
    def _reset(self):
        """ Starting the environment """
        obs = self.env.start(int(self.test))
        return obs

    def _step(self, action):
        """ Next observation """
        obs, reward, terminal, info = self.env.step(action)
        #info = [float(x) for x in str_info.split()[:-1]]
        return obs, reward, terminal, info
        
    def _close(self):
        self.env.fini()
        
    # own
    def set_role(self, test=False):
        self.test = test
        
       
