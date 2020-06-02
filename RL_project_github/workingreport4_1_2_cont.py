# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 20:45:36 2020

@author: khorwei
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from numba import jit

data = [170]
SIZE = 5
HM_EPISODES = 10000000
epsilon = 0.99999999
EPS_DECAY = 0.999999
print((epsilon*EPS_DECAY)**(0.5*HM_EPISODES))
start_q_table = None 
LEARNING_RATE = 0.8 
DISCOUNT = 0.95 
action_range = 21 #range(-1,1,0.1)
limit_switch = 'on'
limit = 0.4
days_limit = 3
stand = 1 # round(0.5*math.sqrt(1/252),2)
mean = 0

if start_q_table is None:
    q_table = {}
    for i in range(SIZE):
        q_table[i] = {}
            

#---------------------------------------------------------------------------------------------------------------------------------------------------
#Function that simulation will untilize


#Generate q_table when needed
def new_state(q_table, state, action_range, weight, POSITION=None):
    if POSITION == None:
        q_table[state]={}
        q_table[state] = np.array([-(np.random.randint(20,40)-0.5)*np.exp(abs(weight))  for i in range(action_range)])##
    else:
        q_table[POSITION]= {}
        q_table[POSITION][state] = {}
        q_table[POSITION][state]= np.array([-(np.random.randint(40, 45)-0.5)*np.exp(abs(weight)) for i in range(action_range)])# higher purnish for big movement
    return 0

@jit(nopython=True)
def fasten(array, begin, control):
    return array[begin:control].max()


# Return maxmimum future value
def max_value_cal(POSITION, q_table, x, Trade_His, SIZE, action_range, days_limit, limit, vol_state, Vol_His):
    limit = int(round(limit*10))
    if POSITION == 0:
        return 0
    else:
        pass
    
    if not Trade_His[x] in q_table[x].keys():
        return Vol_His * (-np.random.randint(20, 40))*round(np.exp(Vol_His),3)
    else:
        list_1 = np.array(list(q_table[x][Trade_His[x]].keys()))
        c1, c2, = vol_state[0] + np.log(1.1), vol_state[0] + np.log(0.9)
        tag = np.logical_and(list_1[:,0] <= c1, list_1[:,0] >= c2)
        list_1 = list_1[tag]
        list_1 = list(map(tuple,list_1))
        if list_1 == []:
            return Vol_His * (-np.random.randint(20, 40))*round(np.exp(Vol_His),3)
        else:
            total = len(list_1)
        
    if x == SIZE-1:
        return sum([q_table[x][Trade_His[-1]][j].max() for j in list_1])/total

    elif POSITION < 0:
        control = int(round(-Trade_His[x]*10))
        end = action_range
        if x < days_limit:
            end = 10 + limit + 1
            if control < 10-limit:
                control = int(10-limit)
            else: pass
        else: pass
        return  sum([fasten(q_table[x][Trade_His[x]][j],control,end) for j in list_1])/total

    else:
        control = int(round(action_range - (Trade_His[x]*10)))
        begin = 0
        if x < days_limit:
            begin = int(round(10 - limit))
            if control > 10 + limit:
                control = int(10+limit+1)
            else: pass
        else: pass
        return  sum([fasten(q_table[x][Trade_His[x]][j],begin,control) for j in list_1])/total

# Given best action to be took and return respective action key
def Action_Finder(Trade_His, i, POSITION, vol_state, q_table, limit, days_limit, action_range, SIZE):
    limit = int(round(limit*10))
    if POSITION < 0 :
        control = int(round(-Trade_His[i]*10))
        end = action_range
        if i < days_limit:
            end = 10 + limit + 1
            if control < 10-limit:
                control = int(10-limit)
            else: pass
        else: pass
        action_key = control + np.argmax(q_table[i][Trade_His[i]][vol_state][control:end])
    else:
        control = int(round(action_range - (Trade_His[i]*10)))
        begin = 0
        if i < days_limit:
            begin = int(round(10 - limit))
            if control > 10 + limit:
                control = int(10+limit+1)
            else: pass
        else: pass
        action_key = begin + np.argmax(q_table[i][Trade_His[i]][vol_state][begin:control])
    return action_key

# Translate selected action into dictionary key
def actionGen(action, cap, flo, i, days_limit):
    if i < days_limit:
        if action > cap:
            action = cap
            action_key = cap * 10 +10
            return round(action, 1),  int(round(action_key))
        elif action < flo:
            action = flo
            action_key = flo * 10 +10
            return round(action, 1),  int(round(action_key))
        else:
            action_key = action * 10 +10
            return round(action, 1),  int(round(action_key))
    action_key = action * 10 +10
    return round(action, 1) , int(round(action_key))

def TriangleDraw(a,b,c,x):
    if x < a:
        return 0
    elif a <= x < c:
        return 2 * (x-a) / ((b-a)*(c-a))
    elif x ==c:
        return 2 / (b-a)
    elif c <= x <= b:
        return - (((b-a)*(b-c)) * x -2*b)
    else:
        return 0 
    

#start simulation
episode_rewards = []
HedgingError = []
Timeforhedge = []
for episode in range(HM_EPISODES):
    episode_reward = 0
    PV = 0
    POSITION = -1
    COST = 0
    OPEN = data[0] #can chg manual input
    Price_His = []
    Trade_His = []
    Vol_His = 0 # need to check
    test = []
    start = time.time()
    c = round((np.random.randint(0,70)-35) / 10,2)
    if limit_switch == 'on':
        cap = limit
        flo = -limit
    else:
        cap = 1
        flo = -1
    for i in range(SIZE):
       # i=0
        if i == 0 :
            ii = 0
            spot = OPEN 
            Price_His.append(spot)
            Trade_His.append(POSITION)
            Vol_His += ii
            test.append(ii)
            vol_state = round(np.log(spot/OPEN),1), round(np.array(test).std(),1)
            
        else: 
            if episode <= HM_EPISODES*0.1 or episode >= HM_EPISODES*0.9:
                sig = round((np.random.randint(0,71)-35)/10*np.sqrt(SIZE/252/SIZE),5) # limit sigma movement (use uniform distribution in training)
                
            elif episode <= HM_EPISODES*0.2 or episode >= HM_EPISODES*0.80:
                sig = round(np.random.normal(0,1)*np.sqrt(SIZE/252/SIZE),5)
                
            else:
                sig = round(np.random.triangular(-3.5, c, 3.5)*np.sqrt(SIZE/252/SIZE),5)
                
            multiplier = np.exp(sig)
            if multiplier > 1.1 or multiplier < 0.9: # Apply trading curb
                if multiplier > 1.1:
                    multiplier = 1.1
                else:
                    multiplier = 0.9
            else:
                multiplier = multiplier
            spot = round(Price_His[-1]*multiplier,4)
            
            ii = round(sig*np.sqrt(252),1)
            Vol_His += abs(ii)
            test.append(ii)
            vol_state = round(np.log(spot/OPEN),1), round(np.array(test).std(),1)
            Price_His.append(spot)
    
        
        if not POSITION in q_table[i].keys() :
            new_state(q_table[i], vol_state, action_range, Vol_His, POSITION=POSITION)
        
        if not vol_state in q_table[i][POSITION].keys() :
            new_state(q_table[i][POSITION], vol_state, action_range, Vol_His)

        if i == SIZE-1:
            action =  -POSITION
            COST += action * spot
            reward = (-abs(COST - OPEN) * 100 + 25) * round(np.exp(abs(Vol_His)),2)  # higher purnish for big movement
            action, action_key = actionGen(action, cap, flo, i, days_limit)
            current_q = q_table[i][POSITION][vol_state][action_key] # Current state
            max_future_q = 0 # Maximum future reward
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q) 
            q_table[i][POSITION][vol_state][action_key] = new_q
            POSITION += action
            episode_reward += reward
            print(episode,i,time.time()-start)
            #print(episode, i, action, POSITION, vol_state, round(reward,2), round(max_future_q,2), round(time.time()-start,2), round(COST,2),round(spot,2), round(epsilon,4) )
            break
        
        if np.random.random() > epsilon: #exploitation
            if episode < HM_EPISODES*0.65:
                if np.random.random() > 0.05: #randomly go to exploration
                    action_key = Action_Finder(Trade_His, i, POSITION, vol_state, q_table, limit, days_limit, action_range, SIZE) 
                else:
                    if POSITION < 0:
                        control = int(round(-POSITION*10))
                        action_key = np.random.randint(control, action_range)
                    else:
                        control = action_range - int(round(POSITION*10))
                        action_key = np.random.randint(control)
            else: #exploitation
                action_key = Action_Finder(Trade_His, i, POSITION, vol_state, q_table, limit, days_limit, action_range, SIZE) 
        else:
            if POSITION < 0:
                control = int(round(-POSITION*10))
                action_key = np.random.randint(control, action_range)
            else:
                control = action_range - int(round(POSITION*10))
                action_key = np.random.randint(control)

        action = round((action_key - 10) / 10, 1)
        action, action_key = actionGen(action, cap, flo, i, days_limit)
        COST += action * spot
        current_q = q_table[i][POSITION][vol_state][action_key]
        POSITION = round(POSITION + action, 1)
        Trade_His.append(POSITION)
                               #punish for over hedge,                                          include diff of perfect hedge position
        reward = 0 - [round(1000*np.exp(Vol_His)*POSITION*spot,2) if POSITION > 0 else 0][0] #-round(abs(POSITION*spot)*100*np.exp(Vol_His),2) 
       # max_future_q = max_value_cal(POSITION, q_table, i+1, Trade_His, SIZE, mean, stand, action_range, days_limit, limit, vol_state)
            
        if POSITION == 0:
            reward += -(abs(COST - OPEN) * 100 + 2.5 )* round(np.exp(abs(Vol_His)),2)  # higher purnish for big movement
            max_future_q = 0
        else:
            max_future_q = round(max_value_cal(POSITION, q_table, i+1, Trade_His, SIZE, action_range, days_limit, limit, vol_state, Vol_His),5)
            pass
        
        if max_future_q < 0: #adjust discount
            max_d = round(1/DISCOUNT,5)
        else:
            max_d = DISCOUNT
            
       # print(time.time()-start)
        new_q = round((1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + max_d * max_future_q),5)
        q_table[i][Trade_His[i]][vol_state][action_key] = new_q
        episode_reward += reward
        
        if POSITION == 0:
            print(episode,i,time.time()-start)
           # print(episode, i, action, POSITION,vol_state,  round(reward,2), round(max_future_q,2), round(time.time()-start,2), round(COST,2),round(spot,2), round(epsilon,4) )
            break

       # print(episode, i, action, POSITION,vol_state,  round(reward,2), round(max_future_q,2), round(time.time()-start,2), round(COST,2),round(spot,2),  round(epsilon,4) )
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
plt.hist(HedgingError[-10000:])
plt.xlabel('HedgingError')
plt.ylabel('Frequency')
plt.xlim(-80,80)
plt.title('HedgingError Distribution')
plt.show()
plt.hist(Timeforhedge[-10000:])
plt.xlabel('Time period / Step taken to hedge')
plt.ylabel('Frequency')
plt.title('Tenor of hedging')
plt.show()


# Output learnt q table 
with open(f"qtable_4_1__2cont.pickle", "wb") as f:
    pickle.dump(q_table, f)
