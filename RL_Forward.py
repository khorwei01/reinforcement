# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 23:42:29 2020

@author: khorwei
"""

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time
import pandas as pd

#Import stock price (Apple-around one month data)
data = pd.read_csv('AAPL.csv')
data = data['Close'].to_numpy()
style.use("ggplot")


SIZE = len(data)
HM_EPISODES = 100000 #Learning times
epsilon = 0.9 #control of Exploration & Exploitation
EPS_DECAY = 0.9998  # Every episode will be epsilon*EPS_DECAY
start_q_table = None # None or Filename
LEARNING_RATE = 0.1 # Speed of learning 
DISCOUNT = 0.95 




if start_q_table is None:
    # initialize the q-table#
    q_table = {}
    for i in range(SIZE):
        for ii in range(2):
            # Give random number to table
            q_table[(i,ii)]=[-np.random.randint(100) for i in range(100)]

else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)


episode_rewards = []

for episode in range(HM_EPISODES):
    episode_reward = 0
    POSITION = -1
    OPEN = data[0] # Shorted price
    for i in range(SIZE):
        # Action in last time step
        if i == SIZE-1:
            reward = -abs(POSITION * (data[i] - OPEN)) #purnish for hedging error
            check = POSITION.copy()
            POSITION -= POSITION
            current_q = q_table[(i, [1 if POSITION < 0 else 0][0])] # Current state
            max_future_q = 0 # Maximum future reward
        
        if POSITION > 0:            
        # Action if in long position
            if np.random.random() > epsilon: 
                # GET THE ACTION
                action = np.argmax(q_table[(i, 0)]) #Exploitation
            else:
                action = np.random.randint(100) #Exploration
            amount = (action + 1) /100
            check = POSITION
            POSITION -= amount
            current_q = q_table[(i, 0)][action]
            reward = -abs(POSITION * (data[i+1] - OPEN))
            max_future_q = np.max(q_table[(i + 1, [1 if POSITION < 0 else 0][0])])

        if POSITION < 0:            
        # Action if in short position 
            if np.random.random() > epsilon:
                # GET THE ACTION
                action = np.argmax(q_table[(i, 1)])
            else:
                action = np.random.randint(100)
            amount = (action + 1) /100
            check = POSITION
            POSITION += amount
            current_q = q_table[(i, 0)][action]
            reward = -abs(POSITION * (data[i+1] - OPEN))
            max_future_q = np.max(q_table[(i + 1, [1 if POSITION < 0 else 0][0])])
        
        # Update current state
        if POSITION == 0:
            reward +=25
            new_q = reward
            
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        
        q_table[(i,  [1 if check < 0 else 0][0])][action] = new_q

        # Update learning rate
        episode_reward += reward
        
        #End episode as target achieved
        if POSITION == 0:
            print(episode, i, reward )
            break

    # Decay the exploration probability
    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY

# Plot learning rate
plt.plot(range(HM_EPISODES), episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Episode Reward')
plt.title('Agent Learning speed')
# Output learnt q table 
#with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
#    pickle.dump(q_table, f)