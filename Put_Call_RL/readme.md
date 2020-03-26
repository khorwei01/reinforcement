1. Option and stock info: 

- Option1.csv and Option 2.csv 
- 1013_1213_AAPL_OPT.csv and 1013_1213_AAPL.csv

2. Put_Call_Parity: 

- pc_parity.py

Agent learnt the best way to replicate a forward is to long an ATM call option and short an ATM put option, given the consideration of premiums. This work excludes the interest and time steps.

3. Put_Call_Parity_Time: 

- pc_t.py (historical timestep + premium + allow to do nothing)

The agent will long an ATM call and short an ATM put at the first time step and do nothing after that.

- cp_t_wp_atm.py (historical timestep + premium + not allow to do nothing)

Agent learnt the best way to replicate a forward given the consideration of premiums. This work includes time step condition (3 time steps and 5 strike). The agent will long an ATM call and short an ATM put at the first time step, then it will enter long and short call (or put) at the ATM strike at each following time step. This work gives no choice to the agent to do nothing. 

- cp_t_wop_atm.py (historical timestep +  x premium + not allow to do nothing)

Agent learnt the best way to replicate a forward given the consideration of premiums. This work includes time step condition (3 time steps and 5 strike). The agent will long an ATM call and short an ATM put at the first time step, then it will enter long and short call (or put) at the same strike at each following time step. This work gives no choice to the agent to do nothing. 

- cp_t_wp_atm_ran.py (random timestamp + premium + not allow to do nothing)

This work deliver the same result as cp_t_wop_atm.py

- cp_t_wp_atm_ran_limit.py (random timestamp + premium + not allow to do nothing + limit unit of option can be traded each timestep )

In progress...


**With the limitation of infinity of the possible value changed of the portfolio, this work only consider the possible of the value changed of portfolio should the agent do a pair trading(long call/put and short put/call with avalaible strike) 

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


Output: qtable-put_chat_time.pickle (pc_t.py)

Overall Learning speed: 

![](https://github.com/khorwei01/reinforcement/blob/master/image/put_call_time.png)
