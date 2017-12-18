#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 16:20:33 2017

@author: Ivan Koryakovskiy <i.koryakovskiy@gmail.com>
"""
import numpy as np
import random
from grlgym.envs.grl import GRLEnv

def main():
  
  # Learning environment
  env = GRLEnv("rbdl_py_balancing.yaml", test=0)
  env.seed(1)
  a = np.zeros(env.action_space.shape) #+ 0.1
  s_fin = run(env, a)
  
  # Testing environment
  eval_env = GRLEnv("rbdl_py_balancing.yaml", test=1)
  s_eval_fin = run(eval_env, a)
    
  # Seeds are different, so result should be different
  assert((s_fin != s_eval_fin).any())

  # Ensure the thread-safetness of the lib
  # Run with reset
  reset_run(env, eval_env, a)

  # Run concurrently  
  concurrent_run(env, eval_env, a)
  
  env.close()
  eval_env.close()
  
  # Testing full environemt
  env = GRLEnv("rbdl_py_balancing_full.yaml", test=1)
  a = np.zeros(env.action_space.shape)
  run(env, a)
  
  # Test environment for many times to catch memory allocation problems if any
  env = GRLEnv("rbdl_py_balancing.yaml", test=1)
  for i in range(1000):
    a = [random.random() for _ in range(env.action_space.shape[0])]
    run(env, a)

def run(e, a):
  s = e.reset()
  s, rw, tt, tau = e.step(a)
  print('%s' % ['%5.5f' % val for val in s])
  
  for i in range(1000):
    s, rw, tt, tau = e.step(a)
    print('env:')
    print('  %s' % ['%5.5f' % val for val in s])
    print('  %s' % [rw, tt, tau])
    if tt:
      break;
  return s 
  
def reset_run(e1, e2, a):
  s1_0 = e1.reset()
  e2.reset()
  for i in range(1000):
    s1_1, rw, tt, tau = e1.step(a)
    e2.reset()
    assert((s1_0 != s1_1).any())
    if tt:
      break;
  
def concurrent_run(e1, e2, a):
  s1_0 = e1.reset()
  s2_0 = e2.reset()
  assert((s1_0 != s2_0).any())
  for i in range(1000):
    s1_1, rw, tt1, tau = e1.step(a)
    s2_1, rw, tt2, tau = e2.step(a)
    assert((s1_1 != s2_1).any())
    if tt1 or tt2:
      break;


if __name__ == "__main__":
  main()