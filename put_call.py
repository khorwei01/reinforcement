# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 15:15:07 2020

@author: khorwei
"""

import numpy as np
import math
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time
import pandas as pd
from itertools import combinations

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
print(check)

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

r = 0
t = 0
spot = data_s['close'][0] # spot price
start_date = data_s['date'][0]
tenor = 0 #location of the tenor in check
strike_list = data_opt1.loc[(data_opt1['date']==data_s['date'][0]) & (data_opt1['mature']==check[tenor]),'strike_price'].unique()  
n = len(strike_list)
post_dict = {0:'Long Call',1:'Short Call',2:'Long Put',3:'Short Put'}


# Create brain 
start_q_table = None
if start_q_table is None:
    # initialize the q-table#
    q_table = {}
    for k in range(2):
        q_table[k] = dict() # long stock, short stock
        for i in range(n*4):
            for j in range(n*4):
                q_table[k][i,j]=-np.random.randint(100,120)
                

LEARNING_RATE = 0.4
DISCOUNT = 0.95
max_future_q = 0
epsilon = 0.999
episode_rewards = []
HM_EPISODES = 20000
EPS_DECAY = 0.99
put_call_parity = [0,1] # 0 equal to long, 1 equal to short

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
    #long call range(1, 1*len(strike_list)+1)
    #short call = range(1*len(strike_list)+1, 2*len(strike_list)+1)
    #long put = range(2*len(strike_list)+1, 3*len(strike_list)+1)
    #short put = range(3*len(strike_list)+1, 4*len(strike_list)+1)
    if np.random.random() > epsilon: 
        # GET THE ACTION
        action = max(q_table[train], key=q_table[train].get) #Exploitation
    else:
        action = np.random.randint(0,len(q_table[train].keys())-1) #Exploration
        action = list(q_table[train])[action]
    
    #payoff of option portfolio
    pay = [payoff(S=close, pick=action[0], choice=len(draw), strike_list=strike_list, call=call, put=put), 
           payoff(S=close, pick=action[1], choice=len(draw), strike_list=strike_list, call=call, put=put)]
    
    trade_hist = [post_dict[action[i]//n] for i in range(len(action))]
    premium += sum([pay[i][1] for i in range(len(pay))])
    reward = [close-spot, spot-close][train] #pnl long, pnl stock
    print((i,reward,sum([pay[i][0] for i in range(len(pay))]), premium) )
    reward = -100*abs(reward-sum([pay[i][0] for i in range(len(pay))]))-100*abs(premium) 
    current_q = q_table[train][action]
    new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
    q_table[train][action] = new_q
    episode_reward = reward
    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY
    
    
plt.plot(range(len(episode_rewards)),episode_rewards)
plt.title('Learning rate')
plt.xlabel('Times of learning')
plt.ylabel('Leanring score')
