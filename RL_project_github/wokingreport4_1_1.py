# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 22:52:21 2020

@author: khorwei
"""


import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
import time
import pandas as pd
import math

#Import stock price (Apple-around one month data)
data = pd.read_csv('Stock_2018.csv')
data = data['close'].to_numpy()

SIZE = 15
HM_EPISODES = 15000 
epsilon = 0.9 
EPS_DECAY = 0.999 
start_q_table = None 
LEARNING_RATE = 0.4 
DISCOUNT = 0.95 
state_range = int(max(data[:SIZE]) - min(data[:SIZE]))
action_range = 21 #range(-1,1,0.1)

if start_q_table is None:
    q_table = {}
    for i in range(SIZE):
        q_table[i] = {}
        for ii in range((i*2*state_range)+1):
            q_table[i][ii] = {}
            for iii in range(action_range):
                q_table[i][ii][iii] = -np.random.randint(100)

else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

def max_valueFinder(POSITION, i, ii, q_table):
    if POSITION < 0 :
        control = int(-POSITION*10)
        key = list(q_table[i][ii].keys())[control:]
        val = list(q_table[i][ii].values())[control:]
        action_key = key[val.index(max(val))]
    else:
        control = action_range - int(POSITION*10)
        key = list(q_table[i][ii].keys())[:control]
        val = list(q_table[i][ii].values())[:control]
        action_key = key[val.index(max(val))]
    return action_key, max(val)
def action_keyGen(action):
    return int(round(action*10+10))

episode_rewards = []

for episode in range(HM_EPISODES):
    episode_reward = 0
    PV = 0
    POSITION = -1
    COST = 0
    OPEN = data[0]
    for i in range(SIZE):
        if i == 0 :
            ii = 0
            ran = 0
        else: 
            ran = np.random.randint(1,SIZE)
            
        if i == SIZE-1:
            action =  -POSITION
            POSITION += action
            COST += action * data[ran]
            reward = -abs(COST - OPEN) * 100 
            action_key = action_keyGen(action)
            current_q = q_table[i][ii][action_key] # Current state
            max_future_q = 0 # Maximum future reward
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[i][ii][action_key] = new_q
            print(episode, i, action, action_key, POSITION )
            break
        if np.random.random() > epsilon:
                action_key, _ = max_valueFinder(POSITION, i, ii, q_table) 
        else:
            if POSITION < 0:
                control = int(-POSITION*10)
                action_key = np.random.randint(control, action_range)
            else:
                control = action_range - int(POSITION*10)
                action_key = np.random.randint(control)
                
        action = round((action_key - 10) / 10, 1)
        COST += action * data[ran]
        POSITION = round(POSITION + action, 1)
        current_q = q_table[i][ii][action_key]
        PV = round(POSITION * math.exp(np.random.normal(0,1)))
        reward = -abs(PV)*100

        if PV > state_range:
            day_end = state_range*2
        elif PV < -state_range:
            day_end = 0
        else:
            day_end = int(PV)+state_range
        
        _, max_future_q = max_valueFinder(POSITION, i+1, day_end, q_table)
         
        if POSITION == 0:
            reward +=25
            reward += -abs(COST - OPEN) * 100
            new_q = reward
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        
        q_table[i][ii][action_key] = new_q
        ii = day_end
        episode_reward += reward
        
        if POSITION == 0:
            #print(episode, i, action, action_key, POSITION )
            break

        #print(episode, i, action, action_key, POSITION )
    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY

# Plot learning rate
plt.plot(range(HM_EPISODES), episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Episode Reward')
plt.title('Agent Learning speed')
# Output learnt q table 
with open(f"qtable_4_1_1.pickle", "wb") as f:
    pickle.dump(q_table, f)