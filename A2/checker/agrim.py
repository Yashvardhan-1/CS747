import numpy as np
import pulp
import sys
import argparse
import sys


def read_mdp(mdp):  # Function to read MDP file

    f = open(mdp)

    Lines = f.readlines()
    n = len(Lines)

    sl = Lines[0].split()
    al = Lines[0].split()
    S = int(sl[1])
    A = int(al[1])

    # Initialize Transition and Reward arrays
    R = np.zeros((S, A, S), dtype=float)
    T = np.zeros((S, A, S), dtype=float)

    for i in range(3, n-2):
        line = Lines[i].split(" ")
        s1, a, s2, r, t = int(line[1]), int(line[2]), int(
            line[3]), float(line[4]), float(line[5])
        R[s1, a, s2] = r
        T[s1, a, s2] = t

    gl = Lines[-1].split()
    gamma = float(gl[1])

    f.close()

    return S, A, R, T, gamma


def read_policy(policy):  # Function to read MDP file

    f = open(policy)

    Lines = f.readlines()
    n = len(Lines)
    ls = []
    for i in range(n):
        ls += [int(Lines[i])]
    return np.array(ls)


def find_v(T, R, gamma, policy):
    V1 = np.zeros(T.shape[0])
    V0 = np.zeros(T.shape[0])

    while(1):
        for s in range(T.shape[0]):
            V1[s] = np.sum(T[s, policy[s], :] * R[s, policy[s], :] +
                           gamma * T[s, policy[s], :] * V0)
        if np.allclose(V1, V0, rtol=1e-13, atol=1e-15):
            break
        else:
            np.copyto(V0, V1)
    return V1


def find_q(V, T, R, gamma):  # Function to find action value function Q
    Q = np.zeros((T.shape[0], T.shape[1]))
    for s in range(T.shape[0]):
        # Find action value for each state action pair
        Q[s] = np.sum(T[s] * R[s] + gamma * T[s] * V, axis=1)
    return Q


def hpi(T, R, gamma):  # Implementation of Howard's PI
    policy = [0 for i in range(T.shape[0])]  # Initialise policy to all zeros
    changed = 1    # Set the flag
    iterations = 0
    while changed == 1:
        iterations += 1
        V = find_v(T, R, gamma, policy)  # Find V
        Q = find_q(V, T, R, gamma)  # Find Q
        improvable_states = []
        for s in range(T.shape[0]):
            if (Q[s][policy[s]] < np.amax(Q[s][:])):
                improvable_states.append(s)
        if len(improvable_states) > 0:
            for k in improvable_states:
                policy[k] = 1 - policy[k]
        else:
            changed = 0
    return V, policy


# numStates 2
# numActions 2
# end -1
# transition 0 0 0 -0.9190312436384449 0.34606241071376004
# transition 0 0 1 0.9309297727238344 0.65393758928624
# transition 0 1 0 -0.28390125061002336 0.6106589110952346
# transition 0 1 1 0.7833213196413649 0.3893410889047654
# transition 1 0 1 0.23673799335066326 1.0
# transition 1 1 0 -0.8024733106817046 1.0
# mdptype continuing
# discount  0.96


def lp_solver(S, A, T, R, gamma):
    prob = pulp.LpProblem('mdp_lp', pulp.LpMinimize)
    decision_variables = pulp.LpVariable.dicts('v', range(S))
    formula = 0.0
    for v in decision_variables.values():
        formula += v

    prob += formula
    for s in range(T.shape[0]):
        for a in range(T.shape[1]):
            formula = 0.0
            for sPrime in range(T.shape[2]):
                formula += (T[s, a, sPrime] * (R[s, a, sPrime] +
                            gamma * decision_variables[sPrime]))
            prob += decision_variables[s] >= formula
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    V = np.array([v.varValue for v in prob.variables()])
    return V


def lp(S, A, T, R, gamma):  # linear programming
    policy = [0 for i in range(T.shape[0])]
    V = lp_solver(S, A, T, R, gamma)
    Q = find_q(V, T, R, gamma)
    for s in range(T.shape[0]):
        if (Q[s][0] < Q[s][1]) and (policy[s] != 1):
            policy[s] = 1

    return V, policy


def print_output(V, pi):
    """Function to print results of any PI Method"""

    for i in range(V.shape[0]):
        print(str.format("{0:.15f}", V[i]) + "\t" + str(pi[i]))

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mdp', type=str)
    parser.add_argument('--algorithm', type=str, default="default")
    parser.add_argument('--policy', type=str, default="default")
    args = parser.parse_args()
    S1, A1, R1, T1, gamma1 = read_mdp(args.mdp)
    if args.policy == "default":

        if args.algorithm == "hpi":
            V_star, pi_star = hpi(T1, R1, gamma1)
        elif args.algorithm == "lp":
            V_star, pi_star = lp(S1, A1, T1, R1, gamma1)

        for v, p in zip(V_star, pi_star):
            print(v, p)
    else:
        p = read_policy(args.policy)
        v = find_v(T1, R1, gamma1, p)

        for v, p in zip(v, p):
            print(v, p)
