#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 16:20:33 2017

@author: Ivan Koryakovskiy <i.koryakovskiy@gmail.com>
"""
import numpy as np
from grlgym.envs.grl import GRLEnv

# Check outcome of policies. It should be different
# if the py_env library is loaded in a separate space
env = GRLEnv("rbdl_py_balancing.yaml", test = 0)

env.seed(1)
s = env.reset()
print('env 0:  %s' % [x for x in s])

eval_env = GRLEnv("rbdl_py_balancing.yaml", test = 1)
se = eval_env.reset()
print('eval_env  0:  %s' % [x for x in se])

a = np.zeros(env.action_space.shape) #+ np.array([0, 0] + [0.1]*7)

for i in range(1000):
  s, rw, tt, tau = env.step(a)
  print('env:')
  print('  %s' % ['%5.5f' % val for val in s])
  print('  %s' % [rw, tt, tau])
  
  se = eval_env.reset()
  
  if tt:
      break;

env.close()
eval_env.close()
