1. Option and stock info: 

Option1.csv and Option 2.csv 

2. Put_Call_Parity: put_call.py

Agent learnt the best way to replicate a forward is to long an ATM call option and short an ATM put option, given the consideration of premiums. This work excludes the interest and time steps.

3. Put_Call_Parity_Time: put_call_time_com.py
Agent learnt the best way to replicate a forward is to long an ATM call option and short an ATM put option, given the consideration of premiums. This work includes time step condition (3 time steps and 5 strike). 
With the limitation of memory, this work mixed the possible trading strategy of replicating a forward but pair option trading in the first step. 

1st step:

This work not considering any position change larger/smaller than (highest K call - lowest K put) in the first steps
example: 

Take 5 strikes, 113-117, 117-113 = 4, 113-117 = -4
4+4+1(take ATM,payoff=0 )

Following step:

This work not considering any position change larger/smaller than existing position state +(highest K call - lowest K put).
example: 

Take 5 strikes, 113-117, 117-113 = 4, 113-117 = -4
9+4+4=17

The Agent managed to choose long ATM call and short ATM put at the first time step and do nth at the following time.

Output: qtable-put_chat_time.pickle

Learning speed: 

![](https://github.com/khorwei01/reinforcement/blob/master/image/put_call_time.png)
