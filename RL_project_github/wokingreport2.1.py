# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 00:12:14 2020

@author: khorwei
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time
import pandas as pd

#Import stock price (Apple-around one month data)
data = pd.read_csv('Stock_2018.csv')
data = data['close'].to_numpy()
style.use("ggplot")


SIZE = 10
HM_EPISODES = 15000 #Learning times
epsilon = 0.999 #control of Exploration & Exploitation
EPS_DECAY = 0.999 # Every episode will be epsilon*EPS_DECAY
start_q_table = None # None or Filename
LEARNING_RATE = 0.4 # Speed of learning 
DISCOUNT = 0.95 
action_range = 11
position_range = 11


if start_q_table is None:
    # initialize the q-table#
    q_table = {}
    for i in range(SIZE):
        q_table[i] = {}
        for ii in range(position_range):
            q_table[i][(ii-10)/10] = {}
            # Give random number to table
            for iii in range(action_range):
                q_table[i][(ii-10)/10][iii/10]=-np.random.randint(50,100) 

else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)


episode_rewards = []

for episode in range(HM_EPISODES):
    episode_reward = 0
    POSITION = -1
    COST = 0
    OPEN = data[0] # Shorted price
    for i in range(SIZE):
        # Action in last time step
        ii = round(POSITION,1)
        if i == SIZE-1:
            reward = -abs(POSITION * (data[i] - OPEN))*1000000 #purnish for hedging error
            action = -POSITION
            POSITION -= POSITION
            current_q = q_table[i][ii][action] # Current state
            max_future_q = 0 # Maximum future reward

        if np.random.random() > epsilon:
            # GET THE ACTION
            key = list(q_table[i][ii].keys())[:int(-POSITION*10+1)]
            val = list(q_table[i][ii].values())[:int(-POSITION*10+1)]
            action = key[val.index(max(val))]
            
        else:
            action = np.random.randint(int(-POSITION*10+1))
            action = action / 10
        
        POSITION += action
        POSITION = round(POSITION,1)
        COST += action * data[i]
        current_q = q_table[i][ii][action]
        reward = -abs(COST - OPEN)*1000000
        
        max_future_q = max([q_table[i][POSITION][round(j/10,1)] for j in range(int(-POSITION*10+1))])
        # Update current state
        if POSITION == 0:
            reward += 25
            
        else: pass
        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        
        q_table[i][ii][action] = new_q

        # Update learning rate
        episode_reward += reward
        
        #End episode as target achieved
        if POSITION == 0:
            break

    # Decay the exploration probability
    print(episode, i, reward )
    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY

# Plot learning rate
plt.plot(range(HM_EPISODES), episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Episode Reward')
plt.title('Agent Learning reward')

# Output learnt q table 
with open(f"qtable_2_1.pickle", "wb") as f:
    pickle.dump(q_table, f)