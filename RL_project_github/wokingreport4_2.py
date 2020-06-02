# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 20:45:36 2020

@author: khorwei
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import time
import math

#---------------------------------------------------------------------------------------------------------------------------------------------------
#Import stock price (Apple-around one month data)
# Initial setup

# May input any price
data_raw = pd.read_csv('AAPL_option.csv')
data_opt = pd.read_csv('AAPL_option_1.csv')

data_opt['strike_price'] = data_opt['strike_price']/1000
data_opt['diff'] = (data_opt['strike_price'] - data_opt['forward_price'])**2
check = data_opt.loc[(data_opt['date']==20141201) & (data_opt['cp_flag'] == 'C') #,: ]
                     & (data_opt['exdate']==20141205),: ].sort_values(by = 'diff')

day_range = len(data_raw.iloc[106:111,:])
hedgetime = 5
HM_EPISODES = 5000000
epsilon = 0.999999
EPS_DECAY = 0.999998#999
(epsilon*EPS_DECAY)**(HM_EPISODES*0.7)
start_q_table = None 
LEARNING_RATE = 0.65 
DISCOUNT = 0.95 
action_range = 21 #range(-1,1,0.1)
limit_switch = 'off'
limit = 0.4
days_limit = 3

if start_q_table is None:
    q_table = {}
    for i in range(hedgetime):
        q_table[i] = {}
        
price = data_raw['close'].values[-41:-21]        # 1 month history
#std = Price_His.std() * np.sqrt(12) / Price_His.mean()    # annualized 1 month std / 1 month mean
implied = check.iloc[0,7] 
PRE = check.iloc[0,6] 
K = check.iloc[0, 4]
#---------------------------------------------------------------------------------------------------------------------------------------------------
#Function that simulation will untilize


#Generate q_table when needed
def new_state(q_table, state, action_range, PRE, cp_flag):
    state_0 = state[0]
    if (PRE >0 and cp_flag == "Call") or (PRE < 0 and cp_flag == "Put"):
        for j in range(11):
            state_1 = round(j/10,1)
            q_table[(state_0, state_1)] = np.random.randint(-450,-400, action_range)
    else:
        for j in range(11):
            state_1 = round((j-10)/10,1)
            q_table[(state_0, state_1)] = np.random.randint(-450,-400, action_range)
    return 0

def t_p(x, tau, mean = 0, sd =1):
    var = sd**2 * tau
    denom = (2*np.pi*var)**.5
    num = np.exp(-(x-mean)**2/(2*var))
    return num/denom

# Return maxmimum future value
def max_value_cal(q_table, tag1, tag2, j, current_state, next_pos, tau, mean):
    c_p = current_state[0]
    key_list = np.array(list(q_table[j+1].keys()))
    key_list = key_list[np.where(key_list[:,1] == next_pos)]
    key_list = list(map(tuple, key_list))
    tp = abs(np.log(np.array([i[0] for i in key_list])/c_p))
    tp = t_p(tp, tau, mean , sd =1)
    tp = tp / sum(tp) 
    return sum([tp[i]*q_table[j+1][key_list[i]][tag1:tag2+1].max() for i in range(len(key_list))])

# Given best action to be took and return respective action key
def Action_Finder(q_table, state, tag1, tag2):
    tag1 = tag1 + 10
    tag2 = tag2 + 10
    val = list(q_table[state][tag1:tag2+1])
    return round((tag1 + val.index(max(val)) - 10) * 0.1, 1), tag1 + val.index(max(val))


def rangeFinder(PRE, POS, cp_flag):
    if (PRE >0 and cp_flag == "Call") or (PRE < 0 and cp_flag == "Put"):
        tag1 = -round(POS*10)
        tag2 = 10 - round(POS*10)
    else:
        tag1 = -round(POS*10) - 10
        tag2 = 10 - round(POS*10) - 10
    return tag1, tag2

def finalSet(PRE, POS, cp_flag, K, spot):
    if (PRE >0 and cp_flag == "Call") or (PRE < 0 and cp_flag == "Put"):
        if spot > K:
            action = round(1 - POS, 1)
            return action, round(action*10 + 10)
        else:
            action = round(0 - POS, 1)
            return action, round(action*10 + 10)
    else:
        if spot > K:
            action = round(- POS - 1, 1)
            return action, round(action*10 + 10)
        else:
            action = round(1 - POS, 1)
            return action, round(action*10 + 10)

def discount_cal(future_val, discount_val):
    if future_val > 0:
        return discount_val
    else:
        return 1/discount_val
    
def cnorm(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def bsPrice(S, r, vol, payoffType, K, T):
    if T == 0:
        if payoffType == 'Call':
            return max(S-K, 0)
        else:
            return max(K-S, 0)
    else:
        fwd = S * math.exp(r * T)
        stdev = vol * math.sqrt(T)
        d1 = math.log(fwd / K) / stdev + stdev / 2
        d2 = d1 - stdev
        if payoffType == 'Call':
            return math.exp(-r * T) * (fwd * cnorm(d1) - cnorm(d2) * K)
        else:
            return math.exp(-r * T) * (K * cnorm(-d2) - cnorm(-d1) * fwd)

#start simulation
episode_rewards = []
HedgingError = []
Timeforhedge = []
for episode in range(HM_EPISODES):
    episode_reward = 0
    cp_flag = 'Call'
    PRE = PRE
    COST = 0
    Price_His = [price[-1], ]
    OPEN =  Price_His[-1] #can chg manual input
    K = K
    POS = 0
    mean = 0
    simula = np.sqrt(1/252*day_range/hedgetime) * np.random.normal(mean,1,hedgetime) 
    std_list = []
    state_list = []
    tag_list = []
    action_list = []
    if limit_switch == 'on':
        cap = limit
        flo = -limit
    else:
        cap = 1
        flo = -1
        
    for i in range(hedgetime):
        start = time.time()
        if i == 0 :
            spot = OPEN 
            state = round(spot,0), POS#, round(np.log(std/std_list[0]),2)
            state_list.append(state)
        else:
            spot = round(Price_His[-1]*np.exp(((-0.5*implied**2) * (1/252) + implied * simula[i-1])),4)
            Price_His = np.concatenate((Price_His, np.array([spot])))
            state = round(spot,0), POS#, round(np.log(std/std_list[0]),2)
            state_list.append(state)
            #PRE = bsPrice(spot, 0, vol, implied, K, (day_range-1)/252*day_range/hedgetime)
            
        if not state in q_table[i].keys() :
            new_state(q_table[i], state, action_range, PRE, cp_flag)
        
        if i == hedgetime-1:
            if spot > K:
                pay = K
            else:
                pay = 0
            action, action_key =  finalSet(PRE, POS, cp_flag, K, spot)
            COST += action * spot
            reward = -abs(COST - PRE - pay) * 1000  + 50
            current_q = q_table[i][state][action_key] # Current state
            max_future_q = 0 # Maximum future reward
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q) 
            q_table[i][state][action_key] = reward
            for j in range(len(state_list)-2,-1,-1):
                current_q = q_table[j][state_list[j]][action_list[j]]
                next_pos = round(state_list[j][1] + round((action_list[j] - 10) /10, 1),1)
                if j < len(state_list)-2:
                    tag1, tag2 = tag_list[j+1]
                    tag1, tag2 = tag1+10, tag2+10
                else:
                    tag1, tag2 = action_key, action_key
                max_future_q = max_value_cal(q_table, tag1, tag2, j, state_list[j], next_pos, 1/252*day_range/hedgetime, mean)
                discount = discount_cal(max_future_q, DISCOUNT)
                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (discount * max_future_q)
                q_table[j][state_list[j]][action_list[j]] = new_q
            episode_reward += reward
            print(episode, i, action, reward)
            break
        else: pass
        
        tag1, tag2 = rangeFinder(PRE, POS, cp_flag)
        tag_list.append((tag1, tag2))        
        
        if np.random.random() > epsilon: #exploitation
            action, action_key = Action_Finder(q_table[i], state, tag1, tag2)
        else:
            action = np.random.randint(tag1,tag2+1)
            action_key = action + 10
            action = round(action * 0.1, 1)
        
        #action, action_key = actionGen(action, cap, flo, i)
        action_list.append(action_key)
        COST += action * spot
        POS += action 
        POS = round(POS,1)
    
        print(episode, i, action, action_key, POS, epsilon )
    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY
    HedgingError.append(COST - OPEN)
    Timeforhedge.append(i)
# Plot learning rate
plt.plot(range(len(episode_rewards)), episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Episode Reward')
plt.title('Agent Learning speed')
plt.show()
plt.hist(HedgingError[-5000:])
plt.xlabel('HedgingError')
plt.ylabel('Frequency')
plt.xlim(-10,10)
plt.hist(HedgingError)
plt.xlabel('HedgingError')
plt.ylabel('Frequency')
plt.xlim(-10,10)
plt.title('HedgingError Distribution')
plt.show()
plt.hist(Timeforhedge)
plt.xlabel('Time period / Step taken to hedge')
plt.ylabel('Frequency')
plt.title('Tenor of hedging')
plt.show()


# Output learnt q table 
with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)
