import argparse
from math import gamma
from pulp import *
from typing import *
import numpy as np

class MDP:
    def __init__(
        self,
        states: int,
        actions: int,
        transitions: np.ndarray,
        rewards: np.ndarray,
        type: str,
        discount: float,
    ) -> None:
        self.states = states
        self.actions = actions
        self.transitions = transitions
        self.rewards = rewards
        self.discount = discount
        self.type = type


def file2mdp(path:str) -> MDP:
    with open(path, "r") as f:
        lines = f.readlines()
        states = int(lines[0].strip().split()[1])
        actions = int(lines[1].strip().split()[1])
        discount = float(lines[-1].strip().split()[1])
        type =  lines[-2].strip().split()[1]
        transition = np.zeros((states,actions,states))
        reward = np.zeros((states,actions,states))

        for line in lines:
            line = line.strip().split()

            if line[0] == "transition":
                state = int(line[1])
                action = int(line[2])
                next_state = int(line[3])
                reward[state, action, next_state] = float(line[4])
                transition[state, action , next_state] = float(line[5])

    return MDP(states, actions, transition, reward, type, discount)

def file2policy(path: str) -> np.ndarray:
    
    if path == None:
        return None
    
    with open(path, "r") as f:
        lines = f.readlines()
        num_lines = len(lines)
        value = np.zeros((num_lines)).astype(int)
        for i, line in enumerate(lines):
            line = line.strip()
            value[i] = int(line)

    return value


def value_iteration(mdp: MDP) -> Tuple[np.ndarray, np.ndarray]:

    values = np.ones((mdp.states))
    policy = np.array([None]*mdp.states)
    while True:

        bellman = np.sum(np.multiply(mdp.transitions , mdp.rewards + mdp.discount*values),axis = 2)
        next_values = np.max(bellman, axis=1)
        policy = np.argmax(bellman,axis=1)
    
        if np.array_equal(values,next_values):
            break
        
        values = next_values

    return values, policy


def policy_iteration(mdp: MDP) -> Tuple[np.ndarray, np.ndarray]:
    
    policy = np.zeros((mdp.states), dtype=int)
    values = np.zeros((mdp.states))

    while True:
        
        values = eval(mdp, policy)
        bellman = np.sum(np.multiply(mdp.transitions , mdp.rewards + mdp.discount*values),axis = 2)
        new_policy = np.argmax(bellman,axis=1)

        if np.array_equal(policy,new_policy):
            break
        
        policy = new_policy

    return values, policy


def linear_programming(mdp: MDP) -> Tuple[np.ndarray, np.ndarray]:
    value, policy = np.zeros((mdp.states)),  np.zeros((mdp.states), dtype =int)

    prob = LpProblem("optimal_v", LpMinimize)
    v = LpVariable.dicts("v", range(mdp.states), lowBound=0, cat="Continuous")

    prob += lpSum(v)

    numS = mdp.states
    # if mdp.type == "episodic" and mdp.discount == 1:
    #     prob += lpSum(v[numS-1]) == 0
    #     numS -= 1

    for s in range(numS):
        for a in range(mdp.actions):
            prob += lpSum([mdp.transitions[s, a, s1] * (mdp.rewards[s, a, s1] + mdp.discount * v[s1]) for s1 in range(mdp.states)]) <= v[s]
    
    prob.solve(PULP_CBC_CMD(msg=0))

    for s in range(mdp.states):
        value[s] = v[s].value()
        policy[s] = np.argmax([np.sum([mdp.transitions[s, a, s1] * (mdp.rewards[s, a, s1] + mdp.discount * v[s1].value()) for s1 in range(mdp.states)]) for a in range(mdp.actions)])

    return value, policy

def eval(mdp: MDP, policy: np.ndarray) -> np.ndarray:

    T = np.zeros((mdp.states,mdp.states))
    Rpi = np.zeros((mdp.states,mdp.states))
    
    for i in range(mdp.states):
        T[i] = mdp.transitions[i,policy[i],:]
        Rpi[i] = mdp.rewards[i,policy[i],:]
    
    R = np.sum(np.multiply(T,Rpi),axis=1)

    values = np.linalg.solve(np.identity(mdp.states) - mdp.discount * T, R)

    return values

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mdp", type=str, required=True, help="Path to the mdp file")
    parser.add_argument(
        "--algorithm",
        type=str,
        required=False,
        default="vi",
        help="The algo to run. Valid values are: vi, hpi, lp",
    )
    parser.add_argument(
        "--policy",
        type=str,
        required=False,
        help="The file containing the policy to be evaluated",
    )

    args = parser.parse_args()

    mdp = file2mdp(args.mdp)
    algorithm = args.algorithm
    policy = file2policy(args.policy)

    if policy is not None:
        assert len(policy) == mdp.states 
        values = eval(mdp,policy)
    else:
        if algorithm == "vi":
            values, policy = value_iteration(mdp)
        elif algorithm == "hpi":
            values, policy = policy_iteration(mdp)
        elif algorithm == "lp":
            values, policy = linear_programming(mdp)

    assert len(policy) == len(values)

    for vali, poli in zip(values, policy):
        print('%.6f'%vali, poli)
