# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 15:15:07 2020

@author: khorwei
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import pandas as pd

#Import stock price (Apple-around one month data)
data_opt = pd.read_csv('Option1.csv') #option quote
data_s = pd.read_csv('Option2.csv') #stock price

#Adjust stock and option data
data_opt1 = data_opt[['date','exdate','strike_price','cp_flag', 'best_bid', 'best_offer' ]]
data_opt1.loc[:,'strike_price'] = data_opt1['strike_price'].values/1000  
data_opt1.loc[:,['date','exdate']] = data_opt1[['date','exdate']].apply(pd.to_datetime, format = '%Y%m%d' )
data_opt1.loc[:, 'mature'] = (data_opt1['exdate'] - data_opt1['date'])
data_s = data_s[['date','close']]
data_s.loc[:,'date'] = data_s['date'].apply(pd.to_datetime, format = '%Y%m%d' )
date_range = len(data_s)

#check tenor of option (in days)
check = data_opt1.loc[data_opt1['date']==data_s.loc[:,'date'][0],'mature'].unique().astype('timedelta64[D]')

r = 0
t = 0
spot = data_s['close'][0] # spot price
start_date = data_s['date'][0]
tenor = 0 #location of the tenor in check
strike_list = data_opt1.loc[(data_opt1['date']==data_s['date'][0]) & (data_opt1['mature']==check[tenor]),'strike_price'].unique()  
n = len(strike_list)

# Create brain 
start_q_table = None
if start_q_table is None:
    # initialize the q-table#
    q_table = {}
    for k in range(2):
        q_table[k] = dict() # long stock, short stock
        for i in range(n*4):
            for j in range(n*4):
                q_table[k][i,j]=-np.random.randint(10,25)     

LEARNING_RATE = 0.4
DISCOUNT = 0.95
max_future_q = 0
epsilon = 0.999
episode_rewards = []
HM_EPISODES = 27000 # better to have all combination (58*4)*(58*4)
EPS_DECAY = 0.99
put_call_parity = [0,1] # 0 equal to long, 1 equal to short

def payoff(S=None, pick = None, choice = None, strike_list = None, call=None, put=None):
    if pick < choice/4 or choice/2 < pick < choice*3/4: 
        pos = 'Long'
        if pick < choice/4:
            option_type = 'C'
        else:
            option_type = 'P'
    else:
        pos = 'Short'
        if pick < choice/2:
            option_type = 'C'
        else:
            option_type = 'P'
    K = strike_list[pick % len(strike_list)]
    if option_type == 'C':
        if pos == 'Long':
            return max(S-K,0), call.loc[call['strike_price']==K,'best_offer'].values[0]*-1
        else:
            return min(K-S,0),  call.loc[call['strike_price']==K,'best_bid'].values[0]
    else:
        if pos == 'Long':
            return max(K-S,0), put.loc[put['strike_price']==K,'best_offer'].values[0]*-1
        else:
            return min(S-K,0), put.loc[put['strike_price']==K,'best_bid'].values[0]
        
for i in range(HM_EPISODES):
    train = put_call_parity[1]
    r = 0
    t = 0
    premium = 0
    sigma = np.random.normal(0,1) # to ensure the close price cover all possible price
    close = spot*math.exp(sigma)
    call = data_opt1.loc[(data_opt1['date']==data_s['date'][0]) 
                         & (data_opt1['mature']==check[0]) 
                         & (data_opt1['cp_flag']=='C'),['strike_price','best_bid','best_offer']].reset_index()
    put = data_opt1.loc[(data_opt1['date']==data_s['date'][0]) 
                         & (data_opt1['mature']==check[0]) 
                         & (data_opt1['cp_flag']=='P'),['strike_price','best_bid','best_offer']].reset_index()
    draw = np.linspace(0,4*n-1,4*n)

    if np.random.random() > epsilon: 
        action = max(q_table[train], key=q_table[train].get) #Exploitation
    else:
        action = np.random.randint(0,len(q_table[train].keys())-1) #Exploration
        action = list(q_table[train])[action]
    
    pay = [payoff(S=close, pick=action[0], choice=len(draw), strike_list=strike_list, call=call, put=put), 
           payoff(S=close, pick=action[1], choice=len(draw), strike_list=strike_list, call=call, put=put)]
    
    premium += sum([pay[i][1] for i in range(len(pay))])
    reward = [close-spot, spot-close][train] #pnl long, pnl stock
    reward = -100*abs(reward-sum([pay[i][0] for i in range(len(pay))]))-100*abs(premium) 
    current_q = q_table[train][action]
    new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
    q_table[train][action] = new_q
    print((i,reward, action, epsilon) )
    episode_reward = reward
    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY
    
    
plt.plot(range(len(episode_rewards)),episode_rewards)
plt.title('Learning rate')
plt.xlabel('Times of learning')
plt.ylabel('Leanring score')

# Output learnt q table 
with open(f"qtable_3_1.pickle", "wb") as f:
    pickle.dump(q_table, f)

