#! /usr/bin/python
import argparse, time
import numpy as np
import pulp as pl

def readtxt(path):

  f = open(path)

  S = int(f.readline().split(" ")[1])
  A = int(f.readline().split(" ")[1])

  end = f.readline().split(" ")
  E = []
  if len(end) == 2:
    E = -1
  else:
    E = end[1:]
  
  R = np.zeros((S, A, S))
  T = np.zeros((S, A, S))

  line = []
  mdpType = ""
  Gamma = 0
  while True:
    line = f.readline().split()
    if len(line) <= 1:
      break;
    if line[0] == "transition":
      s1, a, s2 = int(line[1]), int(line[2]), int(line[3])
      R[s1, a, s2] = float(line[4])
      T[s1, a, s2] = float(line[5])
    elif line[0] == "mdptype":
      mdpType = line[1]
    elif line[0] == "discount":
      Gamma = float(line[1])
      break
    else:
      break
  
  f.close()
  return S, A, R, T, Gamma

def print_mdp(S, A, R, T, gamma):

    """Function to print the data read from MDP file"""

    print ("States: " + str(S))
    print ("Actions: " + str(A))

    print ("Reward Function:")
    for s in range(S):
        for a in range(A):
            for sPrime in range(S):
                print (str(R[s][a][sPrime]) + "\t")
            print ("\n")

    print ("Transition Function:")
    for s in range(S):
        for a in range(A):
            for sPrime in range(S):
                print (str(T[s][a][sPrime]) + "\t")
            print ("\n")
    print ("Gamma: " + str(gamma))
    return

def vi(S, A, R, T, Gamma):

  vprev = np.zeros(S, dtype=float)
  eps = 1e-15
  v = np.zeros(S, dtype=float)
  p = np.zeros(S, dtype=float)

  while True:
    v = np.zeros(S)
    for s in range(S):
      for a in range(A):
        x = np.sum(T[s, a, :] * (R[s, a, :] + Gamma*vprev))
        if x > v[s]:
          v[s] = x
          p[s] = a
    
    if np.sum(np.abs(v-vprev)) < eps:
      break
    else:
      vprev = np.copy(v)
  
  p = pol(S,A,R,T,Gamma,v)
  return v, p 

def pol(S, A, R, T, Gamma, V):
  # T*(R+G*V) 
  # max(a) T(S,A,S)*[R(S,A,S)+ G*V(S)]
  P = np.argmax(np.sum((T[:, :, :])*(R[:,:,:] + Gamma*V), axis = 2), axis = 1)
  return P
  
def get_v(S, A, R, T, Gamma, P):

  model = pl.LpProblem("get_v", pl.LpMinimize)
  var = pl.LpVariable.dicts('v', range(S))

  for s in range(S):
    f = 0.0
    for s1 in range(S):
      f += (T[s, P[s], s1]*(R[s, P[s], s1]+Gamma*var[s1]))

    model += var[s] == f

  model.solve()
  V = np.array([v.varValue for v in model.variables()])
  V = V[1:]
  V = V.T

  return V

def get_q(R, T, Gamma, V):
  return np.sum(T[:,:,:]*(R[:,:,:] + Gamma*V), axis = 2)

def hpi(S, A, R, T, Gamma):
  
  P = np.random.randint(A, size = S)
  print(P)
  # P = np.array([3,3,3,1,4,0,2,0,3,1])
  # Q = SxA matrix

  imp = 1
  V = np.zeros(S)
  eps = 1e-7

  while True:
    V = get_v(S,A,R,T,Gamma,P)
    Q = get_q(R,T,Gamma,V)
    
    improvable = []
    for s in range(S):
      for a in range(A):
        if Q[s,a] >= V[s] and abs(Q[s,a]-V[s]) > eps:
          P[s] = a
          improvable = [s]
    
    if len(improvable) == 0:
      break

  P = pol(S,A,R,T,Gamma,V)
  return V, P

def lp(S, A, R, T, Gamma):
  model = pl.LpProblem("lp_mdp", pl.LpMinimize)
  var = pl.LpVariable.dicts('v', range(S))
  formula = 0.0
  for y in var.values():
    formula += y

  model += formula

  for s in range(S):
    for a in range(A):
      f = 0.0
      for s1 in range(S):
        f += (T[s, a, s1]*(R[s, a, s1]+Gamma*var[s1]))

      model += var[s] >= f

  model.solve()
  
  V = np.array([v.varValue for v in model.variables()])
  P = pol(S,A,R,T,Gamma,V)

  return V, P


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--mdp', type=str, required=True, default = 'NA',help='path to mdp')
  parser.add_argument('--algorithm', type=str, required=True, default = 'lp',help='algorithm mdp')
  parser.add_argument('--policy', type=str, required=False, default = 'NA',help='policy')

  args = parser.parse_args()
  S, A, R, T, Gamma = readtxt(args.mdp)

  V = np.zeros(S)
  P = np.zeros(S)

  if args.algorithm == "vi":
    V, P = vi(S, A, R, T, Gamma)
  elif args.algorithm == "lp":
    V, P = lp(S, A, R, T, Gamma)
  elif args.algorithm == "hpi":
    V, P = hpi(S, A, R, T, Gamma)
  else:
    print("incorrect algo")

  for v, p in zip(V, P):
    print(v, p)
