#! /usr/bin/python
import argparse, time
import numpy as np
from traitlets import Int

def read_p1_para(path):

  f = open(path)

  ls = f.readline().split()
  actions_reward_prob = dict()
  actions_reward_prob[0] = dict()

  at = np.zeros((5,7), dtype=float)

  for i in range(5):
    ls = f.readline().split()
    a = ls[0]
    actions_reward_prob[a] = dict()
    actions_reward_prob[a][-1] = float(ls[1])
    actions_reward_prob[a][0] = float(ls[2])
    actions_reward_prob[a][1] = float(ls[3])
    actions_reward_prob[a][2] = float(ls[4])
    actions_reward_prob[a][3] = float(ls[5])
    actions_reward_prob[a][4] = float(ls[6])
    actions_reward_prob[a][6] = float(ls[7])

    at[i][0] = float(ls[1])
    at[i][1] = float(ls[2])
    at[i][2] = float(ls[3])
    at[i][3] = float(ls[4])
    at[i][4] = float(ls[5])
    at[i][5] = float(ls[6])
    at[i][6] = float(ls[7])

  return at

def read_states(path):
  f = open(path)
  ls = []

  Lines = f.readlines() 
  for line in Lines:
    ls += [int(line)]
  return ls

def mdp(states, actions_reward_prob, q):

  n = len(states)
  a = 5
  d = dict()
  d[1] = dict()
  d[2] = dict()
  
  transition = np.zeros((2*n+1, a, 2*n+1), dtype=float) 
  reward = np.zeros((2*n+1, a, 2*n+1), dtype=float)

  l = 2*n
  
  for i in range(n):
    d[1][states[i]] = int(i)
    d[2][states[i]] = int(i+n)

  for state in states:
    balls = state//100
    runs = state%100
    # for player 1
    s = d[1][state]
    b = balls-1
    for a in range(5):
      x = actions_reward_prob[a]

      if balls%6 == 1:
        
        reward[s, a, l] = 0
        transition[s, a, l] += x[0]or b == 0 

        r = runs
        ns = l if b == 0 else d[2][100*b+r]
        # ns = d[2][100*b+r]
        reward[s, a, ns] = 0
        transition[s, a, ns] += x[1]

        r = runs-1
        # ns = d[1][100*b+r]
        ns = l if r <= 0 or b == 0 else d[1][100*b+r]
        reward[s, a, ns] = 1
        transition[s, a, ns] += x[2]

        r = runs-2
        # ns = d[2][100*b+r]
        ns = l if r <= 0 or b == 0 else d[2][100*b+r]
        reward[s, a, ns] = 2
        transition[s, a, ns] += x[3]

        r = runs-3
        # ns = d[1][100*b+r]
        ns = l if r <= 0 or b == 0 else d[1][100*b+r]
        reward[s, a, ns] = 3
        transition[s, a, ns] += x[4]

        r = runs-4
        # ns = d[2][100*b+r]
        ns = l if r <= 0 or b == 0 else d[2][100*b+r]
        reward[s, a, ns] = 3
        transition[s, a, ns] += x[5]

        r = runs-6
        # ns = d[2][100*b+r]
        ns = l if r <= 0 or b == 0 else d[2][100*b+r]
        reward[s, a, ns] = 3
        transition[s, a, ns] += x[6]
      else:
      
        reward[s, a, l] = 0
        transition[s, a, l] += x[0]

        r = runs
        ns = l if b == 0 else d[1][100*b+r]
        # ns = d[1][100*b+r]
        reward[s, a, ns] = 0
        transition[s, a, ns] += x[1]

        r = runs-1
        ns = l if r <= 0 or b == 0 else d[2][100*b+r]
        # ns = d[2][100*b+r] 
        reward[s, a, ns] = 1
        transition[s, a, ns] += x[2]

        r = runs-2
        ns = l if r <= 0 or b == 0 else d[1][100*b+r]
        # ns = d[1][100*b+r]
        reward[s, a, ns] = 2
        transition[s, a, ns] += x[3]

        r = runs-3
        ns = l if r <= 0 or b == 0 else d[2][100*b+r]
        # ns = d[2][100*b+r]
        reward[s, a, ns] = 3
        transition[s, a, ns] += x[4]


        r = runs-4
        ns = l if r <= 0 or b == 0 else d[1][100*b+r]
        # ns = d[1][100*b+r]
        reward[s, a, ns] = 3
        transition[s, a, ns] += x[5]


        r = runs-6
        ns = l if r <= 0 or b == 0 else d[1][100*b+r]
        # ns = d[1][100*b+r]
        reward[s, a, ns] = 3
        transition[s, a, ns] += x[6]

    # for player 2
    s = d[2][state]
    
    if balls%6 == 1:
      reward[s, 0, l] = 0
      transition[s, 0, l] = q

      r = runs
      ns = l if b == 0 else d[1][100*b+r]
      # ns = d[1][100*b+r]   
      reward[s, 0, ns] = 0
      transition[s, 0, ns] = (1-q)/2

      r = runs-1
      # ns = d[2][100*b+r]  
      ns = l if r <= 0 or b == 0 else d[2][100*b+r] 
      reward[s, 0, ns] = 1
      transition[s, 0, ns] = (1-q)/2
    else:
      reward[s, 0, l] = 0
      transition[s, 0, l] = q

      r = runs
      ns = l if b == 0 else d[2][100*b+r]
      # ns = d[2][100*b+r]   
      reward[s, 0, ns] = 0
      transition[s, 0, ns] = (1-q)/2

      r = runs-1
      # ns = d[1][100*b+r]  
      ns = l if r <= 0 or b == 0 else d[1][100*b+r] 
      reward[s, 0, ns] = 1
      transition[s, 0, ns] = (1-q)/2
  
  return transition, reward, 2*n+1

def write_mdp(transition, rewards, numStates):
  print(f"numStates {numStates}")
  print(f"numActions 5")
  print("episodic")

  for i in range(numStates):
    for a in range(5):
      for j in range(numStates):
        print("transition", i, a, j, rewards[i,a,j], transition[i,a,j])
  
  print("mdptype episodic")
  print("discount  1")
  
  

  pass
    
if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('--states', type=str, required=False, default = 'statefilepath.txt',help='statefilepath')
  parser.add_argument('--p1_parameters', type=str, required=False, default = './data/cricket/sample-p1.txt',help='p1_parameters')
  parser.add_argument('--q', type=str, required=False, default = 'NA',help='q for player 2')

  args = parser.parse_args()
  trans = read_p1_para(args.p1_parameters)
  states = read_states(args.states)
  q = float(args.q)

  transition, rewards, numStates = mdp(states, trans, q)
  write_mdp(transition, rewards, numStates)
  
  

  