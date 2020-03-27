# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 14:36:47 2020

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
HM_EPISODES = 30000 #Learning times
epsilon = 0.99 #control of Exploration & Exploitation
EPS_DECAY = 0.9998  # Every episode will be epsilon*EPS_DECAY
start_q_table = None # None or Filename
LEARNING_RATE = 0.7 # Speed of learning 
DISCOUNT = 0.98 
limit = 0.2
days_limit = 3
ss = max(abs(data[:SIZE]-data[0]))
state_range = int(2*ss+1)
action_range = 20 + 1
r = 0.0
r = [np.exp(r*i/360) for i in range(SIZE)]
commission = 0.0
ran ='on'
limit_switch = 'on'

#START A BRAIN
if start_q_table is None:
    q_table = {}
    for i in range(SIZE): 
        q_table[i] = dict()
        for ii in range(state_range): #LIMIT RANGE OF POS CHG
            q_table[i][ii]= dict()
            for iii in range(action_range): # LIMIT THE CHOICE RANGE(-1,1,0.1)
                q_table[i][ii][iii] = -np.random.randint(100,150)# Give random number to table
            

#else:
#    with open(start_q_table, "rb") as f:
#        q_table = pickle.load(f)
def actionGen(action, cap, flo, state, i, days_limit):
    
    if  state < 0 and (-1 > action + state or action + state> 0):
        action = -state
    elif state > 0 and (0 > action + state or action + state > 1):
        action = -state
    else:
        pass
    if i < days_limit:
        if action > cap:
            action = cap
            action_key = cap * 10 +10
            return round(action, 1), round(action_key)
        
        elif action < flo:
            action = flo
            action_key = flo * 10 +10
            return round(action, 1), round(action_key)
        
        else:
            action_key = action * 10 +10
            return round(action, 1), round(action_key)
        
    action_key = action * 10 +10
    return round(action, 1) , round(action_key)

episode_rewards = []
er = []

for episode in range(HM_EPISODES):
    episode_reward = 0
    state = -1
    OPEN = data[0]*r[-1] # Shorted price
    hed_cost = []
    trade_hist = []
    for i in range(SIZE):
        if i != 0 and ran == 'on':
            time = np.random.randint(1,SIZE)
        else:
            time = i
        
        if limit_switch == 'on':
            cap = limit
            flo = -limit
        else:
            cap = 1
            flo = -1
            
        ii = int((data[time]-data[0]) +  ss)# INDICATE CURRENT STATE 
        
        # Action in last time step
        if i == SIZE-1 or abs(state) < 10e-3:
            action = -state 
            action, action_key = actionGen(action, cap, flo, state, i, days_limit-1)
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
                
                if len(trade_hist)-1 == j and j == i:
                    max_future_q = nx[4]
                elif i < days_limit and abs(action) < limit:
                    if action > 0:
                        s_range = [q_table[nx[0]][nx[1]][10+k-int(limit*10)] for k in range(int(10+action*10)+1)]
                        max_future_q = max(s_range)
                    else:
                        s_range = [q_table[nx[0]][nx[1]][10+k-int(limit*10)] for k in range(int(10+action*10), int(limit*10+10))]
                        max_future_q = max(s_range)
                else:
                    s_range = [q_table[nx[0]][nx[1]][10+k-int(limit*10)] for k in range(int(2*limit*10)+1)]
                    max_future_q = max(s_range)
                
                if max_future_q < 0:
                    DISCOUNT_dyna = 1/DISCOUNT
                else:
                    DISCOUNT_dyna = DISCOUNT
                q_table[th[0]][th[1]][th[2]] = (1-LEARNING_RATE) * th[3] + LEARNING_RATE * (th[4] + DISCOUNT_dyna * max_future_q)
            
            if i < days_limit:
                er.append('error' + str(episode) + str(i))
            else:
                print(episode, i, ii, action_key, reward, epsilon)
            break

        if state > 0:# Action if in long position
            if np.random.random() > epsilon: 
                # GET THE ACTION
                key = np.argmax([q_table[i][ii][10 + k] for k in range(int(-state*10),11)])
                action_key = key #EXPLOITATION
                action = (action_key - 10) / 10
                
            else:
                action_key = np.random.randint(int(-state*10), 11) #Exploration
                action = (action_key - 10) / 10

        if state < 0:            
            if np.random.random() > epsilon: 
                # GET THE ACTION
                key = np.argmax([q_table[i][ii][10 + k] for k in range(int(state*10),11)])
                action_key = key + 10 #EXPLOITATION
                action = (action_key - 10) / 10
                
            else:
                action_key = np.random.randint(int(state*10), 21) #Exploration
                action = (action_key - 10) / 10
       
        # IN FORCE LIMIT
        action, action_key = actionGen(action, cap, flo, round(state,1), i, days_limit) 
        com = commission * abs(action)
        current_q = q_table[i][ii][action_key]
        reward = 0
        state = state + action
        hed_cost.append( action * data[time] + com)
        trade_hist.append((i, ii, action_key, current_q, reward))
        print(episode, i, ii, action_key, reward, epsilon)
        # Update learning rate
        episode_reward += reward
        
            
    # Decay the exploration probability
    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY

# Plot learning rate
plt.scatter(range(len(episode_rewards)), episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Episode Reward')
plt.title('Agent Learning speed')
# Output learnt q table 
#with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
#    pickle.dump(q_table, f)