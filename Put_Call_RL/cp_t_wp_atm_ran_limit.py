# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 14:31:54 2020

@author: khorwei
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from datetime import datetime, timedelta

#Import stock price (Apple-around one month data)
data_opt = pd.read_csv('1013_1213_AAPL_OPT.csv') #option quote
data_s = pd.read_csv('1013_1213_AAPL.csv') #stock price


#Adjust stock and option data
data_opt1 = data_opt[['date','exdate','strike_price','cp_flag', 'best_bid', 'best_offer','forward_price' ]]
data_opt1 = data_opt1.loc[data_opt1['forward_price']>0,:]
data_opt1.loc[:,'strike_price'] = data_opt1['strike_price'].values/1000  
data_opt1.loc[:,['date','exdate']] = data_opt1[['date','exdate']].apply(pd.to_datetime, format = '%Y%m%d' )


data_s = data_s[['date','close']]
data_s.loc[:,'date'] = data_s['date'].apply(pd.to_datetime, format = '%Y%m%d' )

date_range = 3

#PAYOFF COMPUTER
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

# FIND TENOR
def matureFinder(x,tenor, unit, data_opt1):
    tenor = np.timedelta64(tenor, unit)
    mature = x + tenor
    exdate = data_opt1.loc[data_opt1['date']== x, 'exdate'].unique()
    error = np.argmin([abs((pd.Timestamp(exdate[i])-mature)/timedelta(days=1)) for i in range(len(exdate))])
    return exdate[error]

# FIND NEAREST ATM STRIKE OPTION
def strikeListGen(spot_date, daily_price, data_opt1, num, ex_date):
    strike_list = data_opt1.loc[(data_opt1['date']==spot_date) & (data_opt1['exdate']==ex_date),'strike_price'].unique()
    strike_list = np.sort(strike_list)
    idx = int(np.argmin([(i - daily_price)**2 for i in strike_list]))
    strike_list = strike_list[idx-num:idx+1+num]
    return strike_list


r = 0
t = 0
spot = data_s['close'][0] # spot price
start_date = data_s['date'][0]
tenor = 1
unit = 'M'
numOfStrike = 3
choice =  numOfStrike*4

date_range = 3 #ONLY THREE DAY, MANY DAY THEN COM INCREASE GEO

# Create brain, consider only long position
start_q_table = None
if start_q_table is None:
    # initialize the q-table#
    q_table = {}
    for k in range(date_range):
        q_table[k] = dict() # time step
        for l in range(2*(k+1)*(numOfStrike-1)+1): #COVER ONLY RANGE OF OPTION PAIR TRADE
            q_table[k][l] = dict()  
            for i in range(choice): # 0-4 long call at diff K, 5 = do nth
                for j in range(choice): # 0-4 short put at diff K, 5 = do nth 
                    q_table[k][l][i,j]=-np.random.randint(20,50) #INITIAL A LARGE NUM TO ACCESS ALL POSSIBLE PICK


LEARNING_RATE = 0.7
DISCOUNT = 0.99
epsilon = 0.9999
episode_rewards = []
HM_EPISODES = 8000 # better to have all combination (36*1)*(36*17)*(36*25)
EPS_DECAY = 0.999
put_call_parity = [0,1] # 0 equal to long, 1 equal to short
ran_date_range = len(data_s)-31 #BE CAREFUL OF TENOR RANGE, MAY NOT ABLE TO SEARCH DATA NOT IN RANGE

for i in range(HM_EPISODES):
    train = put_call_parity[0] #SIMULATE LONG POS ONLY 
    r = 0
    t = 0
    premium = []
    sigma = np.random.normal(0,0.1) # to ensure the close price cover all possible price
    close = spot*math.exp(sigma)
    last = close-spot #long position only
    payoffLast = []
    trade_hist = []
    for j in range(date_range):
        #PROVIDE DISCRETE RANDOM CHOICE OF SPOT, TRY TO AVOID OVERFITTING
        if j != 0:
            sigma = np.random.normal(0,0.1) # to ensure the close price cover all possible price
            close = spot*math.exp(sigma)
            tod_idx = np.random.randint(1, ran_date_range)
        else:
            tod_idx = j
            
        #GET OPTION INFO 
        ex_date =  matureFinder(data_s['date'][tod_idx], tenor, unit , data_opt1)
        strike_list = strikeListGen(data_s['date'][tod_idx], data_s['close'][tod_idx], data_opt1, int((numOfStrike-1)/2), ex_date)       
        call = data_opt1.loc[(data_opt1['date']==data_s['date'][tod_idx]) 
                             & (data_opt1['exdate']==ex_date) 
                             & (data_opt1['cp_flag']=='C'),['strike_price','best_bid','best_offer']].reset_index()
        put = data_opt1.loc[(data_opt1['date']==data_s['date'][tod_idx]) 
                             & (data_opt1['exdate']==ex_date) 
                             & (data_opt1['cp_flag']=='P'),['strike_price','best_bid','best_offer']].reset_index()
        
        #COMPUTE STATE OF POSITION (DISCRETE)
        last = round(last)
        lower = -(numOfStrike-1)*(j+1)
        upper = (numOfStrike-1)*(j+1)
        
        #GET POSITION STATE
        if j==0:
            l=0
        else:
            l = 0 if last < lower else 2*upper if last > upper else last+upper
        
        # GET THE ACTION
        if np.random.random()  > epsilon: #
            action = max(q_table[j][l], key=q_table[j][l].get) #Exploitation
        else:
            action = np.random.randint(0,len(q_table[j][l].keys())) #Exploration
            action = list(q_table[j][l])[action]
        
        #PAYOFF OF ACTION
        pay = [payoff(S=close, pick=action[0], choice=choice, strike_list=strike_list, call=call, put=put), 
               payoff(S=close, pick=action[1], choice=choice, strike_list=strike_list, call=call, put=put)]
        
        #RECORD PAYOFF &  PREMIUM
        payoffLast.append(round(sum([pay[i][0] for i in range(len(pay))]), 2))
        premium.append( sum([pay[i][1] for i in range(len(pay))]))
        
        #REWARD OF EACH PERIOD
        if j == date_range-1:
            last = last - payoffLast[j] - abs(premium[j])
            reward = -1000*abs(last) - 1000*abs(sum(premium)) 
            trade_hist.append((j, action, l, reward, 2*upper))
            q_table[j][l][action] = reward
            his = len(trade_hist)
            
            #UPDATE STATE VALUE OF EACH CHOICE BASED ON FINAL PAYOFF
            for i in range(his-1,0,-1):
                nx = trade_hist[i]
                th = trade_hist[i-1]
                q_table[th[0]][th[2]][th[1]] = th[3] * (1-LEARNING_RATE) + LEARNING_RATE *max(q_table[nx[0]][nx[2]].values()) # max([max(q_table[nx[0]][nx[2]].values()) for t in range(nx[4])])#

        else:
            reward = 0
            
            #IF FIRST TIME DO NTH, LONG POSITION SHALL CHG (TO RECONSIDER IT IS NECESSARY)
            if j==0 and action == (numOfStrike,numOfStrike):
                close = spot*math.exp(sigma)
                last = round(close-spot)
            else:
                last = round(last - payoffLast[j] - abs(premium[j]))
            current_q = q_table[j][l][action]
            trade_hist.append((j, action,l, current_q, 2*upper)) #RECORD TRDE INFO
            
        print((i,j,action, reward,  close, last, payoffLast[j], premium[j], l, epsilon))
        
    episode_reward = reward
    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY
    
    
plt.plot(range(len(episode_rewards)),episode_rewards)
plt.title('Learning rate')
plt.xlabel('Times of learning')
plt.ylabel('Leanring score')

#Output qtable
#with open(f"q_table_ran.pickle", "wb") as f:
#    pickle.dump(q_table, f)