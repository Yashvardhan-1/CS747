"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the AlgorithmManyArms class. Here are the method details:
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
"""

import sqlite3
import numpy as np

# START EDITING HERE
# You can use this space to define any helper functions that you need

def tomp(succ, fail):
    return np.random.beta(succ+1, fail+1)

vec_tomp = np.vectorize(tomp)

# END EDITING HERE

class AlgorithmManyArms:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
        # Horizon is same as number of arms
        # print(f"{num_arms} and {horizon}")
        self.arm_to_sample = 0
        # self.succ = np.zeros(int(np.sqrt(num_arms)))
        # self.fail = np.zeros(int(np.sqrt(num_arms)))
        # self.fac = int(np.sqrt(num_arms))
        self.succ = np.zeros(num_arms)
        self.fail = np.zeros(num_arms)
    
    def give_pull(self):
        # START EDITING HERE
        return self.arm_to_sample
        # tomp = vec_tomp(self.succ, self.fail)
        # return np.argmax(tomp)*self.fac
        # raise NotImplementedError
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        if reward == 0:
            self.fail[arm_index] += 1
        else:
            self.succ[arm_index] += 1
        
        if self.succ[arm_index]/(self.succ[arm_index]+ self.fail[arm_index]) < 0.97 : 
            self.arm_to_sample = np.random.randint(self.num_arms)
        # raise NotImplementedError
        # END EDITING HERE
