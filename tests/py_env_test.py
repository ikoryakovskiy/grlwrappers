#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 16:20:33 2017

@author: Ivan Koryakovskiy <i.koryakovskiy@gmail.com>
"""
import numpy as np

from grlgym.envs.grl import GRLEnv
env = GRLEnv("rbdl_py_balancing.yaml")

s = env.reset()

print([x for x in s])

a = np.zeros(env.action_space.shape) #+ np.array([0, 0] + [0.1]*7)

for i in range(1000):
  s, rw, tt, tau = env.step(a)
  print(['%5.5f' % val for val in s])
  print((rw, tt, tau))
  if tt:
      break;

env.close()
