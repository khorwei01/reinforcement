1. RL_Forward.py

The agent managed to learn hedge a forward immediately after a shorted position.

2. RL_Forward_ran_limit.py

With the element of random walk and limit unit of option trade each time, the agent perform as following:

Given 15 time steps, random market price, only able to long/short maximum of 0.5 stock each time step, short 1 unit of forward.

Action:
long/short unit of stock in each time step

0,0.5,0.4,0.5,-0.5,0.5,-0.5,0.5,-0.5,0.5,-0.5,0.5,-0.5,0.5,-0.4

with the consideration of commission and interest rate

0,0.5,0.2,0.5,-0.5,0.5,-0.5,0.5,-0.5,0.5,-0.5,0.5,-0.5,0.5,0.2

** Randomness confuses the agent to form a strategy, then it only make a minor change in the hedging action. Clearly the agent need more information in this case.
