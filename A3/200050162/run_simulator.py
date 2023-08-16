from importlib.resources import path
from gym_driving.assets.car import *
from gym_driving.envs.environment import *
from gym_driving.envs.driving_env import *
from gym_driving.assets.terrain import *

import time
import pygame, sys
from pygame.locals import *
import random
import math
import argparse

# Do NOT change these values
TIMESTEPS = 1000
FPS = 30
NUM_EPISODES = 10

class Task1():

    def __init__(self):
        """
        Can modify to include variables as required
        """

        super().__init__()

    def next_action(self, state):
        """
        Input: The current state
        Output: Action to be taken
        TO BE FILLED
        """

        # Replace with your implementation to determine actions to be taken
        action_steer = None
        action_acc = None

        """
            state = np.array([self.x, self.y, self.vel, self.angle])
        """ 
        
        action_acc = 2
        action_steer = 1

        y1 = 0-state[1]
        x1 = 350-state[0]
        rad = np.pi*(state[3]/180)
        cosA_B = np.cos(rad)*x1/(np.sqrt(x1**2+y1**2)) + np.sin(rad)*y1/(np.sqrt(x1**2+y1**2))

        if cosA_B < 0.97:
            action_steer = 2
        else:
            action_acc = 4

        # if state[1]<-25:
        #     if abs(state[3]-90) > 3:
        #         action_steer = 2
        #     else:
        #         action_acc = 1
        #         action_acc = 4
        # elif state[1] > 25:
        #     if abs(state[3]-270) > 3:
        #         action_steer = 2
        #     else:
        #         action_acc = 1
        #         action_acc = 4
        # else:
        #     # make horizontal and move
        #     if state[3] > 3 and state[3] < 357:
        #         if state[3] > 180:
        #             action_steer = 2
        #         else:
        #             action_steer = 0
        #     else:
        #         action_steer = 1
        #         action_acc = 3

        action = np.array([action_steer, action_acc])  
        return action

    def controller_task1(self, config_filepath=None, render_mode=False):
        """
        This is the main controller function. You can modify it as required except for the parts specifically not to be modified.
        Additionally, you can define helper functions within the class if needed for your logic.
        """
    
        ######### Do NOT modify these lines ##########
        pygame.init()
        fpsClock = pygame.time.Clock()

        if config_filepath is None:
            config_filepath = '../configs/config.json'

        simulator = DrivingEnv('T1', render_mode=render_mode, config_filepath=config_filepath)

        time.sleep(3)
        ##############################################

        # e is the number of the current episode, running it for 10 episodes
        for e in range(NUM_EPISODES):
        
            ######### Do NOT modify these lines ##########
            
            # To keep track of the number of timesteps per epoch
            cur_time = 0

            # To reset the simulator at the beginning of each episode
            state = simulator._reset()
            
            # Variable representing if you have reached the road
            road_status = False
            ##############################################

            # The following code is a basic example of the usage of the simulator
            for t in range(TIMESTEPS):
        
                # Checks for quit
                if render_mode:
                    for event in pygame.event.get():
                        if event.type == QUIT:
                            pygame.quit()
                            sys.exit()

                action = self.next_action(state)
                state, reward, terminate, reached_road, info_dict = simulator._step(action)
                fpsClock.tick(FPS)

                cur_time += 1

                if terminate:
                    road_status = reached_road
                    break

            # Writing the output at each episode to STDOUT
            print(str(road_status) + ' ' + str(cur_time))

class Task2():

    def __init__(self):
        """
        Can modify to include variables as required
        """

        super().__init__()
        self.ran_cen_list = None

    def next_action(self, state):
        """
        Input: The current state
        Output: Action to be taken
        TO BE FILLED

        You can modify the function to take in extra arguments and return extra quantities apart from the ones specified if required
        """

        # Replace with your implementation to determine actions to be taken
        action_steer = None
        action_acc = None

        action_acc = 2
        action_steer = 1

        """
            state = np.array([self.x, self.y, self.vel, self.angle])
            x,y
            1,1 is lower left
            2,2 is upper left 
            3,3 is lower right
            4,4 is upper right
        """ 

        th = 65
        th_safe_y = 40
        direct = self.ran_cen_list
        safe_y_p = min(direct[0][1], direct[2][1])-th
        safe_y_p = min(safe_y_p, th_safe_y)
        safe_y_n = max(direct[1][1], direct[3][1])+th 
        safe_y_n = max(safe_y_n, -th_safe_y)

        if state[1] > safe_y_n and state[1] < safe_y_p:
            if state[3] > 3 and state[3] < 357:
                if state[3] > 180:
                    action_steer = 2
                else:
                    action_steer = 0
            else:
                action_acc = 4
        elif(
            (state[0]>0 and state[1]<0 and (abs(direct[1][0]-state[0]) > th or direct[1][1] < state[1]) ) or
            (state[0]<0 and state[1]>0 and (abs(direct[2][0]-state[0]) > th or direct[2][1] > state[1]) ) or
            (state[0]>0 and state[1]>0 and (abs(direct[0][0]-state[0]) > th or direct[0][1] > state[1]) ) or
            (state[0]<0 and state[1]<0 and (abs(direct[2][0]-state[0]) > th or direct[3][1] < state[1]) )
        ):
            if state[1] > 0:
                if abs(state[3]-270) > 5:
                    if state[3]-90 > 0 and state[3] < 270:
                        action_steer = 2
                    else:
                        action_steer = 0
                else:
                    action_acc = 4
            else:
                if abs(state[3]-90) > 5:
                    if state[3] > 90 and state[3] < 270:
                        action_steer = 0
                    else:
                        action_steer = 2
                else:
                    action_acc = 4
        else:
            direction = "right"
            if(
                (state[0]>0 and state[1]>0 and state[0] < direct[0][0] and abs(direct[0][1]-state[1]) > th) or
                (state[0]>0 and state[1]<0 and state[0] < direct[1][0] and abs(direct[1][1]-state[1]) > th) or
                (state[0]<0 and state[1]>0 and state[0] < direct[2][0] and abs(direct[2][1]-state[1]) > th) or
                (state[0]<0 and state[1]<0 and state[0] < direct[3][0] and abs(direct[2][1]-state[1]) > th)
            ):
                direction = "left"
            
            if direction == "right":
                if state[3] > 3 and state[3] < 357:
                    if state[3] > 180:
                        action_steer = 2
                    else:
                        action_steer = 0
                else:
                    action_acc = 4
            else:
                if abs(state[3]-180) > 5:
                    if state[3] > 180:
                        action_steer = 0
                    else:
                        action_steer = 2
                else:
                    action_acc = 4
        
        # safe_v3 = min(direct[2][0], direct[3][0])-th
        # safe_v2_l = max(direct[2][0], direct[3][0])+th
        # safe_v2_r = min(direct[0][0], direct[1][0])-th
        # safe_v1 = max(direct[0][0], direct[1][0])+th

        # if state[0]>0 and state[1]>0:
        #     x_cen, y_cen = direct[0][0], direct[0][1] 
        #     if abs(x_cen-state[0]) > th:

        # elif state[0]>0 and state[1]<0:
        #     x_cen, y_cen = direct[1][0], direct[1][1]
        # elif state[0]<0 and state[1]>0:
        #     x_cen, y_cen = direct[2][0], direct[2][1]
        # else:
        #     x_cen, y_cen = direct[3][0], direct[3][1]

        # elif (state[0] < safe_v3) or (state[0] > safe_v2_l and state[0] < safe_v2_r) or (state[0]>safe_v1):
        #     if state[1] > 0:
        #         if abs(state[3]-270) > 5:
        #             if state[3]-90 > 0 and state[3] < 270:
        #                 action_steer = 2
        #             else:
        #                 action_steer = 0
        #         else:
        #             action_acc = 4
        #     else:
        #         if abs(state[3]-90) > 5:
        #             if state[3] > 90 and state[3] < 270:
        #                 action_steer = 0
        #             else:
        #                 action_steer = 2
        #         else:
        #             action_acc = 4
        # else:
        #     if state[0]>0 and state[1]>0:
        #         x_cen, y_cen = direct[0][0], direct[0][1] 
        #         if abs(x_cen-state[0]) > th:

        #     elif state[0]>0 and state[1]<0:
        #         x_cen, y_cen = direct[1][0], direct[1][1]
        #     elif state[0]<0 and state[1]>0:
        #         x_cen, y_cen = direct[2][0], direct[2][1]
        #     else:
        #         x_cen, y_cen = direct[3][0], direct[3][1]


        # if state[1]<-25:
        #     if abs(state[3]-90) > 3:
        #         action_steer = 2
        #     else:
        #         action_acc = 1
        #         action_acc = 4
        # elif state[1] > 25:
        #     if abs(state[3]-270) > 3:
        #         action_steer = 2
        #     else:
        #         action_acc = 1
        #         action_acc = 4
        # else:
        #     # make horizontal and move
        #     if state[3] > 3 and state[3] < 357:
        #         if state[3] > 180:
        #             action_steer = 2
        #         else:
        #             action_steer = 0
        #     else:
        #         action_steer = 1
        #         action_acc = 3

        action = np.array([action_steer, action_acc])  

        return action

    def controller_task2(self, config_filepath=None, render_mode=False):
        """
        This is the main controller function. You can modify it as required except for the parts specifically not to be modified.
        Additionally, you can define helper functions within the class if needed for your logic.
        """
        
        ################ Do NOT modify these lines ################
        pygame.init()
        fpsClock = pygame.time.Clock()

        if config_filepath is None:
            config_filepath = '../configs/config.json'

        time.sleep(3)
        ###########################################################

        # e is the number of the current episode, running it for 10 episodes
        for e in range(NUM_EPISODES):

            ################ Setting up the environment, do NOT modify these lines ################
            # To randomly initialize centers of the traps within a determined range
            ran_cen_1x = random.randint(120, 230)
            ran_cen_1y = random.randint(120, 230)
            ran_cen_1 = [ran_cen_1x, ran_cen_1y]

            ran_cen_2x = random.randint(120, 230)
            ran_cen_2y = random.randint(-230, -120)
            ran_cen_2 = [ran_cen_2x, ran_cen_2y]

            ran_cen_3x = random.randint(-230, -120)
            ran_cen_3y = random.randint(120, 230)
            ran_cen_3 = [ran_cen_3x, ran_cen_3y]

            ran_cen_4x = random.randint(-230, -120)
            ran_cen_4y = random.randint(-230, -120)
            ran_cen_4 = [ran_cen_4x, ran_cen_4y]

            ran_cen_list = [ran_cen_1, ran_cen_2, ran_cen_3, ran_cen_4]            
            eligible_list = []

            self.ran_cen_list = ran_cen_list
            # To randomly initialize the car within a determined range
            for x in range(-300, 300):
                for y in range(-300, 300):

                    if x >= (ran_cen_1x - 110) and x <= (ran_cen_1x + 110) and y >= (ran_cen_1y - 110) and y <= (ran_cen_1y + 110):
                        continue

                    if x >= (ran_cen_2x - 110) and x <= (ran_cen_2x + 110) and y >= (ran_cen_2y - 110) and y <= (ran_cen_2y + 110):
                        continue

                    if x >= (ran_cen_3x - 110) and x <= (ran_cen_3x + 110) and y >= (ran_cen_3y - 110) and y <= (ran_cen_3y + 110):
                        continue

                    if x >= (ran_cen_4x - 110) and x <= (ran_cen_4x + 110) and y >= (ran_cen_4y - 110) and y <= (ran_cen_4y + 110):
                        continue

                    eligible_list.append((x,y))

            simulator = DrivingEnv('T2', eligible_list, render_mode=render_mode, config_filepath=config_filepath, ran_cen_list=ran_cen_list)
        
            # To keep track of the number of timesteps per episode
            cur_time = 0

            # To reset the simulator at the beginning of each episode
            state = simulator._reset(eligible_list=eligible_list)
            ###########################################################

            # The following code is a basic example of the usage of the simulator
            road_status = False

            for t in range(TIMESTEPS):
        
                # Checks for quit
                if render_mode:
                    for event in pygame.event.get():
                        if event.type == QUIT:
                            pygame.quit()
                            sys.exit()

                action = self.next_action(state)
                state, reward, terminate, reached_road, info_dict = simulator._step(action)
                fpsClock.tick(FPS)

                cur_time += 1

                if terminate:
                    road_status = reached_road
                    break

            print(str(road_status) + ' ' + str(cur_time))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="config filepath", default=None)
    parser.add_argument("-t", "--task", help="task number", choices=['T1', 'T2'])
    parser.add_argument("-r", "--random_seed", help="random seed", type=int, default=0)
    parser.add_argument("-m", "--render_mode", action='store_true')
    parser.add_argument("-f", "--frames_per_sec", help="fps", type=int, default=30) # Keep this as the default while running your simulation to visualize results
    args = parser.parse_args()

    config_filepath = args.config
    task = args.task
    random_seed = args.random_seed
    render_mode = args.render_mode
    fps = args.frames_per_sec

    FPS = fps

    random.seed(random_seed)
    np.random.seed(random_seed)

    if task == 'T1':
        
        agent = Task1()
        agent.controller_task1(config_filepath=config_filepath, render_mode=render_mode)

    else:

        agent = Task2()
        agent.controller_task2(config_filepath=config_filepath, render_mode=render_mode)
