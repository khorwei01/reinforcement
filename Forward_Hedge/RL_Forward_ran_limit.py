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
import pandas as pd

#Import stock price (Apple-around one month data)
data = pd.read_csv('Stock_2018.csv')
data = data['close'].to_numpy()
style.use("ggplot")




SIZE = 15
HM_EPISODES = 25000 #Learning times
epsilon = 0.9 #control of Exploration & Exploitation
EPS_DECAY = 0.9998  # Every episode will be epsilon*EPS_DECAY
start_q_table = None # None or Filename
LEARNING_RATE = 0.9 # Speed of learning 
DISCOUNT = 0.95 
SIZE = 15
limit = 0.5
state_range = 20+1
r = 0.3
r = [np.exp(r*i/360) for i in range(SIZE)]
commission = 0.01
ran ='off'
limit_switch = 'on'

#START A BRAIN
if start_q_table is None:
    q_table = {}
    for i in range(SIZE): 
        q_table[i] = dict()
        for ii in range(state_range): #LIMIT RANGE OF POS CHG
            q_table[i][ii]= dict()
            for iii in range(state_range): # LIMIT THE CHOICE RANGE(-1,1,0.1)
                q_table[i][ii][iii] = -np.random.randint(100,150)# Give random number to table
            

#else:
#    with open(start_q_table, "rb") as f:
#        q_table = pickle.load(f)
def actionGen(action, cap, flo):
    if action > cap:
        action = cap
        action_key = cap * 10 +10
        return action, round(action_key)
    elif action < flo:
        action = flo
        action_key = flo * 10 +10
        return action, round(action_key)
    else:
        action_key = action * 10 +10
        return action, round(action_key)

episode_rewards = []

for episode in range(HM_EPISODES):
    episode_reward = 0
    state = -1
    OPEN = data[0] # Shorted price
    hed_cost = []
    trade_hist = []
    for i in range(SIZE):

        if i != 0 and ran == 'on':
            time = np.random.randint(1,15)
        else:
            time = i
        
        if limit_switch == 'on':
            cap = limit
            flo = -limit
        else:
            cap = 1
            flo = -1
            
        ii = round(state * 10) + 10 # INDICATE CURRENT STATE 
        
        # Action in last time step
        if i == SIZE-1 or abs(state) < 10e-3:
            action = -state 
            action, action_key = actionGen(action, cap, flo)
            state -=action
            com = commission * abs(action)
            hed_cost.append( action * data[time] + com)
            reward = -abs(sum(hed_cost*np.flip(r)[:i+1]) - OPEN)*1000 + 25 - abs(state*99999999)#purnish for hedging error
            current_q = q_table[i][ii][action_key] # Current state
            max_future_q = 0 # Maximum future reward
            q_table [i][ii][action_key] = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            trade_hist.append((i, ii, action_key, current_q, reward))
            episode_reward += reward
            
            for j in range(len(trade_hist)-1,0,-1):
                th = trade_hist[j-1]
                nx = trade_hist[j]
                if len(trade_hist)-1 == j:
                    max_future_q = nx[4]
                else:
                    s_range = [q_table[i][ii][10+k-int(limit*10)] for k in range(int(2*limit*10)+1)]
                    max_future_q = max(s_range)
                q_table[th[0]][th[1]][th[2]] = (1-LEARNING_RATE) * th[3] + LEARNING_RATE * (th[4] + DISCOUNT * max_future_q )
            
            print(episode, i, ii, action_key, reward, epsilon)
            break

        if state > 0:            
        # Action if in long position
            if np.random.random() > epsilon: 
                # GET THE ACTION
                key = np.argmax([q_table[i][ii][k] for k in range(0,11)])
                action_key = key #EXPLOITATION
                action = (action_key - 10) / 10
                
            else:
                action_key = np.random.randint(0, 11) #Exploration
                action = (action_key - 10) / 10

        if state < 0:            
            if np.random.random() > epsilon: 
                # GET THE ACTION
                key = np.argmax([q_table[i][ii][k] for k in range(10,21)])
                action_key = key + 10 #EXPLOITATION
                action = (action_key - 10) / 10
                
            else:
                action_key = np.random.randint(10, 21) #Exploration
                action = (action_key - 10) / 10
                
        # IN FORCE LIMIT
        action, action_key = actionGen(action, cap, flo)
        com = commission * abs(action)
        current_q = q_table[i][ii][action_key]
        reward = 0
        state += action
        hed_cost.append( action * data[time] + com)
        trade_hist.append((i, ii, action_key, current_q, reward))
        print(episode, i, ii, action_key, reward, epsilon)
        # Update learning rate
        episode_reward += reward
        
            
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
