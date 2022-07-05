import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
import random
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from operator import add
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
plt.rcParams["figure.figsize"] = (20,15)

class GridEnvironment(gym.Env):
    
    def __init__(self, type):                     #type variable defines the environment type ie deterministic or stochastic
        self.observation_space = spaces.Discrete(25)
        self.action_space = spaces.Discrete(4)
        self.max_timesteps = 25
        self.reward_action_step = -0.5         #reward associated with each timestep
        self.outbound_reward = -1             #each time the agent tries to go out of the environment it gets negative reward
        self.env_type = type
        print('***************{}***************'.format(self.env_type+' environment'))
        
    def reset(self):
        self.timestep = 0
        self.reward = 0
        self.done = False
        self.agent_pos = [0,0]
        self.intermediate_goal1 = [2,1]
        self.intermediate_goal2 = [0,4]
        self.intermediate_goal3 = [4,0]
        self.final_goal_pos = [4,3]
        self.monster_pos = [2,3]
        self.pit_pos = [3,1]
        self.state = np.zeros((5, 5))
        
        self.state[tuple(self.agent_pos)] = 1
        observation = self.state.flatten()
        self.reward_dict = {                         #dict to store the values of rewards
            'intermediate_goal1': 2,
            'intermediate_goal2': 0.5,
            'intermediate_goal3': 2,
            'final_goal': 25,
            'monster': -10,
            'pit': -5
        }
        self.visited_dict = {                        #dict to store the visited status 
            'intermediate_goal1': 0,
            'intermediate_goal2': 0,
            'intermediate_goal3': 0,
            'final_goal_pos': 0,
            'monster_pos': 0,
            'pit_pos': 0
        }
        
        return observation
    
    def step(self,action):

        current_reward = self.reward_action_step
        self.timestep += 1
        
        #defining the stochastic part of the environment
        if(self.env_type == 'stochastic'):
            if(action == 0):
                action = np.random.choice(4, 1, p=[0.95, 0.03, 0.01, 0.01])[0]
            
            elif(action == 1):
                action = np.random.choice(4, 1, p=[0.03, 0.95, 0.01, 0.01])[0]
            
            elif(action == 2):
                action = np.random.choice(4, 1, p=[0.01, 0.01, 0.95, 0.03])[0]
            
            else:
                action = np.random.choice(4, 1, p=[0.01, 0.01, 0.03, 0.95])[0]
    
        old_pos = self.agent_pos.copy()
        
        if action == 0:
            self.agent_pos[1] -= 1 #down
        
        if action == 1:
            self.agent_pos[1] += 1 #up
        
        if action == 2:
            self.agent_pos[0] -= 1 #left
        
        if action == 3:
            self.agent_pos[0] += 1 #right
        
        if(self.agent_pos[0] > 4 or self.agent_pos[1] > 4 or self.agent_pos[0] < 0 or self.agent_pos[1] < 0):
            current_reward += self.outbound_reward
        
        self.agent_pos = np.clip(self.agent_pos, 0, 4)         #clip function to ensure safety of the agent
        
        self.state = np.zeros((5,5))
        
        next_action_possible = 1                            #defines whether the next action is possible or not
                                                            # 1 if possible 0 if goal reached and -1 if dead by monster or in pit
        
        if(self.agent_pos == self.intermediate_goal1).all():
            if(self.visited_dict['intermediate_goal1'] == 0):
                current_reward += self.reward_dict['intermediate_goal1']
            self.visited_dict['intermediate_goal1'] += 1
            
        if(self.agent_pos == self.intermediate_goal2).all():
            
            if(self.visited_dict['intermediate_goal2'] == 0):
                current_reward += self.reward_dict['intermediate_goal2']
            
            self.visited_dict['intermediate_goal2'] += 1
            
        if(self.agent_pos == self.intermediate_goal3).all():
            
            if(self.visited_dict['intermediate_goal3'] == 0):
                current_reward += self.reward_dict['intermediate_goal3']
            
            self.visited_dict['intermediate_goal3'] += 1
        
        if (self.agent_pos == self.final_goal_pos).all():
            current_reward += self.reward_dict['final_goal']
            next_action_possible = 0
            self.visited_dict['final_goal_pos'] += 1
        
        self.done = True if(self.timestep >= self.max_timesteps or (self.agent_pos == self.final_goal_pos).all())else False
        
        if(self.agent_pos == self.monster_pos).all():
            self.visited_dict['monster_pos'] += 1
            current_reward += self.reward_dict['monster']
            self.done = True
            next_action_possible = -2
        
        if(self.agent_pos == self.pit_pos).all():
            self.visited_dict['pit_pos'] += 1
            current_reward += self.reward_dict['pit']
            self.done = True
            next_action_possible = -1
        
        self.reward += current_reward
        info = {'next_action_possible': next_action_possible, 'current_agent_pos': self.agent_pos}
        self.state = np.zeros((5, 5))
        self.state[tuple(self.agent_pos)] = 1
        observation = self.state.flatten()
        return observation, current_reward, self.done, info
    
    def action_values(self, action):                                      #method to give the name of the action
        a = ''
        if(action == 0):
            a = 'Down'
        if(action == 1):
            a = 'Up'
        if(action == 2):
            a = 'Left'
        if(action == 3):
            a = 'Right'
        return a
    
    def render(self):
        fix ,ax = plt.subplots(figsize=(10,10))
        ax.set_xlim(0,5)
        ax.set_ylim(0,5)

        agent = AnnotationBbox(OffsetImage(plt.imread('./images/agent.png'), zoom=0.20),  # Plotting the agent.
                       list(map(add, self.agent_pos, [0.5, 0.5])), frameon=False)
        if(self.agent_pos[0] == self.intermediate_goal1[0] and self.agent_pos[1] == self.intermediate_goal1[1]):
            if(self.visited_dict['intermediate_goal1'] == 1):
                agent = AnnotationBbox(OffsetImage(plt.imread('./images/agent_reward.png'), zoom=0.15),  # Plotting the agent with reward.
                           list(map(add, self.agent_pos, [0.5, 0.5])), frameon=False)
        elif(self.agent_pos[0] == self.intermediate_goal3[0] and self.agent_pos[1] == self.intermediate_goal3[1]):
            if(self.visited_dict['intermediate_goal3'] == 1):
                agent = AnnotationBbox(OffsetImage(plt.imread('./images/agent_reward.png'), zoom=0.15),  # Plotting the agent with reward.
                           list(map(add, self.agent_pos, [0.5, 0.5])), frameon=False)
        elif(self.agent_pos[0] == self.intermediate_goal2[0] and self.agent_pos[1] == self.intermediate_goal2[1]):
            if(self.visited_dict['intermediate_goal2'] == 1):
                agent = AnnotationBbox(OffsetImage(plt.imread('./images/agent_reward.png'), zoom=0.15),  # Plotting the agent with reward.
                           list(map(add, self.agent_pos, [0.5, 0.5])), frameon=False)
        elif(self.agent_pos[0] == self.pit_pos[0] and self.agent_pos[1] == self.pit_pos[1]):
            agent = AnnotationBbox(OffsetImage(plt.imread('./images/agent_in_pit.png'), zoom=0.08),  # Plotting the agent in pit.
                       list(map(add, self.agent_pos, [0.5, 0.5])), frameon=False)
        elif(self.agent_pos[0] == self.final_goal_pos[0] and self.agent_pos[1] == self.final_goal_pos[1]):
            agent = AnnotationBbox(OffsetImage(plt.imread('./images/agent_goal.png'), zoom=0.18),  # Plotting the agent with goal.
                       list(map(add, self.agent_pos, [0.5, 0.5])), frameon=False)
        elif(self.agent_pos[0] == self.monster_pos[0] and self.agent_pos[1] == self.monster_pos[1]):
            agent = AnnotationBbox(OffsetImage(plt.imread('./images/dead_agent.png'), zoom=0.18),  # Plotting the dead agent.
                       list(map(add, self.agent_pos, [0.5, 0.5])), frameon=False)
        ax.add_artist(agent)
        
        pit = AnnotationBbox(OffsetImage(plt.imread('./images/pit.png'), zoom=0.10),  # Plotting the pit.
                       list(map(add, self.pit_pos, [0.5, 0.5])), frameon=False)
        if(self.visited_dict['pit_pos'] == 0):
            ax.add_artist(pit)
        
        monster = AnnotationBbox(OffsetImage(plt.imread('./images/monster.png'), zoom=0.12),  # Plotting the monster.
                       list(map(add, self.monster_pos, [0.5, 0.4])), frameon=False)
        if(self.visited_dict['monster_pos'] == 0):
            ax.add_artist(monster)
        
        goal = AnnotationBbox(OffsetImage(plt.imread('./images/goal.png'), zoom=0.12),  # Plotting the goal.
                       list(map(add, self.final_goal_pos, [0.5, 0.5])), frameon=False)
        if(self.visited_dict['final_goal_pos'] == 0):
            ax.add_artist(goal)
        
        intermediate_goal1 = AnnotationBbox(OffsetImage(plt.imread('./images/small_reward.png'), zoom=0.09),  # Plotting the intermediate reward.
                       list(map(add, self.intermediate_goal1, [0.5, 0.5])), frameon=False)
        if(self.visited_dict['intermediate_goal1'] == 0):
            ax.add_artist(intermediate_goal1)
        
        
        intermediate_goal3 = AnnotationBbox(OffsetImage(plt.imread('./images/small_reward.png'), zoom=0.09),  # Plotting the intermediate reward.
                       list(map(add, self.intermediate_goal3, [0.5, 0.5])), frameon=False)
        if(self.visited_dict['intermediate_goal3'] == 0):
            ax.add_artist(intermediate_goal3)
        
        intermediate_goal2 = AnnotationBbox(OffsetImage(plt.imread('./images/big_reward.png'), zoom=0.09),  # Plotting the intermediate reward.
                       list(map(add, self.intermediate_goal2, [0.5, 0.5])), frameon=False)
        if(self.visited_dict['intermediate_goal2'] == 0):
            ax.add_artist(intermediate_goal2)
        
        plt.grid()
        plt.show()