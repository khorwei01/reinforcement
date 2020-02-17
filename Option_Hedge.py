# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 10:42:45 2020

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

def stdev(x):
    avg = sum(x)/len(x)
    return round((sum([(i-avg)**2 for i in x])/len(x))**(1/2))
#Import stock price (Apple-around one month data)
data = pd.read_csv('Option_test.csv', header = None)
#data = data['Close'].to_numpy()
data = data[0].to_numpy()
style.use("ggplot")


SIZE = len(data)
HM_EPISODES = 1000000 #Learning times
epsilon = 0.9999 #control of Exploration & Exploitation
EPS_DECAY = 0.99999  # Every episode will be epsilon*EPS_DECAY
#start_q_table = None # None or Filename
LEARNING_RATE = 0.3 # Speed of learning 
DISCOUNT = 0.99 




#if start_q_table is None:
    # initialize the q-table#
#    q_table = {}
#    for i in range(SIZE):#ime step
#        for ii in range(2):# Long or short position on physcal stock
#            for iii in range(10000): #price diff in basis point from Ytd price
#                for iv in range(10000):#price diff in basis point from K
#                # Give random number to table
#                q_table[(i,ii,iii, iv)]=[-np.random.randint(100) for i in range(100)]
q_table = np.random.randint(low=-100, high = 0, size = (SIZE, 2, 201), dtype = 'int64')

#trial[2,1,10, 10,action]
#else:
#    with open(start_q_table, "rb") as f:
#        q_table = pickle.load(f)

POSITION=0

episode_rewards = []
i=6
for episode in range(HM_EPISODES):
    episode_reward = 0
    K=175
    P=0
    C=5.85
    POSITION=0
    COST=-5.85
    path=[]
    #OPEN = data[0] # Shorted price
    for i in range(SIZE):
        ii = [1 if POSITION > 0 else 0][0]
        #iii = int(round((data[i]-K)/data[i]*100/2+50-1))
        #iv = int(stdev(data[:i+1]))
        upper_bound = int(min((1-POSITION)*100 + 100 + 1, 201))
        lower_bound = int(max((1-POSITION)*100 + 1 , 0))

        #First into the position
        if i == 0:
            if C > 0:
                ii = 1 # receive premium Short Call, delta hedging long stock
                if P > 0:
                    ii = 0 # receive premium Short Put, delta hedging short stock
                else:
                    ii = 1 #Pay premium Long Put
            else:
                ii = 0 #Pay premium Long Call
            if np.random.random() > epsilon: 
                # GET THE ACTION
                action = np.argmax(q_table[i, ii, lower_bound:upper_bound]) #Exploitation
                action = action + lower_bound
            else:
                action = np.random.randint(low = 100, high = 201) #Exploration
            amount = (action -100)/100
            POSITION += amount
            COST += amount*data[i]
            current_q = q_table[i, ii, action]
            if data[i] > K:
                reward = [0 if POSITION < 0.5 else 50][0] 
            else:
                reward = [50 if POSITION < 0.5 else 0][0] 
            max_future_q = np.max(q_table[i+1, [1 if POSITION > 0 else 0][0], :])
        
        # Action in last time step        
        elif i == SIZE-1:
            if data[i]>K:
                action = int((1 - POSITION) * 100 + 100)
                POSITION += 1 - POSITION
                amount = (action -100)/100
                COST = COST + amount * data[i] - K 
                reward = -abs(COST*1000)
            else:
                action = int(-POSITION * 100 + 100 )
                POSITION += 0 - POSITION
                amount = (action - 100)/100
                COST = COST + amount * data[i] 
                reward = -abs(COST*1000)
            current_q = q_table[i, ii, action] # Current state
            #check = POSITION.copy()
            POSITION -= POSITION 
            max_future_q = 0 # Maximum future reward
            new_q = reward
             
        elif data[i] > K :            
        # Action if in long position
            if np.random.random() > epsilon: 
                # GET THE ACTION
                action = np.argmax(q_table[i, ii, lower_bound:upper_bound]) #Exploitation
                action = action + lower_bound
            else:
                action = np.random.randint(low = lower_bound, high = upper_bound) #Exploration
            amount = (action -100)/100
            #print('buy', POSITION, action)
            POSITION += amount
            COST += amount*data[i]
            reward = [-10 if POSITION < 0.5 else 10][0]
            #check = POSITION
            current_q = q_table[i, ii, action ]
            max_future_q = np.max(q_table[i+1, [1 if  POSITION > 0 else 0][0], :]) # Avoid forward looking

        elif data[i] < K:            
        # Action if in short position 
            if np.random.random() > epsilon:
                # GET THE ACTION
                action = np.argmax(q_table[i, ii, lower_bound:upper_bound])
                action = action + lower_bound
            else:
                action = np.random.randint(low = lower_bound, high = upper_bound)
            #check = POSITION
            amount = (action -100)/100
            #print('sell', POSITION, action )
            POSITION += amount
            COST += amount*data[i]
            current_q = q_table[i, ii, action ]
            reward = [-10 if POSITION > 0.5 else 10][0]
            max_future_q = np.max(q_table[i+1, [1 if POSITION > 0 else 0][0], :])
                # Update current state
            
        path.append((POSITION, amount, COST))
        if i == SIZE-1:
            new_q = reward
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            
        q_table[i,  [1 if POSITION > 0 else 0][0], action] = new_q

        # Update learning rate
        episode_reward += reward

        #End episode as target achieved
        if i == SIZE-1:
            print(episode, round(reward,2), round(epsilon,2), round(COST,2), round(POSITION,2) )
            break

    # Decay the exploration probability
    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY

# Plot learning rate
plt.plot(range(1303764), episode_rewards)

# Output learnt q table 
#with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
#    pickle.dump(q_table, f)

