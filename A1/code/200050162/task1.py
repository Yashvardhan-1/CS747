"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the base Algorithm class that all algorithms should inherit
from. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)

We have implemented the epsilon-greedy algorithm for you. You can use it as a
reference for implementing your own algorithms.
"""

import numpy as np
import math
from scipy.optimize import fsolve
# Hint: math.log is much faster than np.log for scalars

class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
    
    def give_pull(self):
        raise NotImplementedError
    
    def get_reward(self, arm_index, reward):
        raise NotImplementedError

# Example implementation of Epsilon Greedy algorithm
class Eps_Greedy(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # Extra member variables to keep track of the state
        self.eps = 0.1
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
    
    def give_pull(self):
        if np.random.random() < self.eps:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)
    
    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value


# START EDITING HERE
# You can use this space to define any helper functions that you need
def UCB(e_mean, step, numPlays):
   return e_mean + math.sqrt(2 * math.log(step + 1) / numPlays)

vec_UCB = np.vectorize(UCB)

#   kl(x,y)=xlog(x/y)+(1−x)log((1−x)/(1−y))
# def KLdiv(y, x):
#     return x*math.log(x/y) + (1-x)*math.log((1-x)/(1-y))

def KL(p, q):
	if p == 1:
		return p*np.log(p/q)
	elif p == 0:
		return (1-p)*np.log((1-p)/(1-q))
	else:
		return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))


def KLucb(e_mean, steps, pulls, c=3):
    l = e_mean
    r = 1
    eps = 1e-3
    temp = e_mean
    qold = e_mean
    q = l + (r-l)/2
    rhs = (math.log(steps) + c*math.log(math.log(steps)))/pulls
    while(abs(qold-q)>eps):
        temp = KL(e_mean, q)
        if temp > rhs:
            r = q
        elif temp == 0:
            break
        else:
            l = q
        qold = q
        q = l + (r-l)/2
    return float(q)

vec_KL_UCB = np.vectorize(KLucb)

# def solve_q(rhs, e_mean):
# 	if e_mean == 1:
# 		return 1 
# 	q = np.arange(e_mean, 1, 0.01)
# 	lhs = []
# 	for el in q:
# 		lhs.append(KL(e_mean, el))
# 	lhs_array = np.array(lhs)
# 	lhs_rhs = lhs_array - rhs
# 	lhs_rhs[lhs_rhs <= 0] = np.inf
# 	min_index = lhs_rhs.argmin()
# 	return q[min_index]
    

def tomp(succ, fail):
    return np.random.beta(succ+1, fail+1)

vec_tomp = np.vectorize(tomp)

# END EDITING HERE

class UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.steps = 1*num_arms
        self.e_mean = np.zeros(num_arms)
        self.pulls = np.ones(num_arms)
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        ucb = vec_UCB(self.e_mean, self.steps, self.pulls)
        return np.argmax(ucb)
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.steps += 1
        self.pulls[arm_index] += 1
        n = self.pulls[arm_index]
        e_mean = self.e_mean[arm_index]
        self.e_mean[arm_index] = ((n - 1) / n) * e_mean + (1 / n) * reward 
        pass
        # END EDITING HERE

class KL_UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.steps = 1*num_arms
        self.e_mean = np.zeros(num_arms)
        self.pulls = np.ones(num_arms)
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        # l = []
        # for i in range(self.num_arms):
        #     l += [KLucb(self.e_mean[i], self.steps, self.pulls[i])]
        l = vec_KL_UCB(self.e_mean, self.steps, self.pulls)
        return np.argmax(np.array(l))
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.steps += 1
        self.pulls[arm_index] += 1
        n = self.pulls[arm_index]
        e_mean = self.e_mean[arm_index]
        self.e_mean[arm_index] = ((n - 1) / n) * e_mean + (1 / n) * reward 
        pass
        # END EDITING HERE


class Thompson_Sampling(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.succ = np.zeros(num_arms)
        self.fail = np.zeros(num_arms)

        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        tomp = vec_tomp(self.succ, self.fail)
        return np.argmax(tomp)
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        if reward == 0:
            self.fail[arm_index] += 1
        else:
            self.succ[arm_index] += 1
        pass
        # END EDITING HERE
