#! /usr/bin/python
import argparse, time
import numpy as np

def read_states(path):
  f = open(path)
  ls = []

  Lines = f.readlines() 
  for line in Lines:
    ls += [int(line)]
  return ls

def decoder(states, path):
  f = open(path)
  d = {0:0, 1:1, 2:2, 3:4, 4:6}

  for s in states:
    c = f.readline().split()
    b = int(s)//100
    r = int(s)%100
    print(str(b).zfill(2) + str(r).zfill(2), d[int(c[1])], c[0])

  # for s in states:
  #   c = f.readline().split()
  #   print(s, d[int(c[1])], c[0])

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('--value-policy', type=str, required=False, default = 'value_and_policy_file.txt',help='value-policy')
  parser.add_argument('--states', type=str, required=False, default = 'statefilepath.txt',help='state_file_path')
  
  args = parser.parse_args()
  
  states = read_states(args.states)
  decoder(states, args.value_policy)

  