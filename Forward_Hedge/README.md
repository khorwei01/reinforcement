1. RL_Forward.py

The agent managed to learn hedge a forward immediately after a shorted position.

2. RL_Forward_ran_limit_buy_only.py

Action policy: agent only allow to buy stock when shorted forward.
With the element of random walk and limit unit of option trade each time, the agent perform as following:

Example 1:

Given 15 time steps, random market price, only able to long maximum of 0.2 stock for 5 time step, short 1 unit of forward.

Agent will hedge 0.2 unit of stock at the forst 5 time step.

With the consideration of interest rate and commission, the result is consistent

Example 2

Given 15 time steps, random market price, only able to long maximum of 0.2 stock for 3 time step, short 1 unit of forward.

Agent will hedge 0.2 unit of stock at the forst 3 time step and hedge the rest of it.

With the consideration of interest rate and commission, the result is consistent

3. RL_Forward_ran_limit_mixed.py

Action policy: agent only allow to buy or sell stock (condition of 0 \leq stock position \leq 1) when shorted forward.
With the element of random walk and limit unit of option trade each time, the agent's action :

Example 1:

Given 15 time steps, random market price, only able to long/short maximum of 0.2 stock for the first 5 time step, short 1 unit of forward.

Agent will hedge 0.2 unit of stock at the first 5 time step.

With the consideration of interest rate and commission, the result is consistent

Example 2

Given 15 time steps, random market price, only able to long maximum of 0.2 stock for 3 time step, short 1 unit of forward.

Agent will hedge the forward position according to the profit or lose of the portfolio, no simple consistent strategy is observe. We need to observe more and built more detail environment to test the agent.


** Note discount on future value subject to change according the the positive or negative of maximum future value. A wrong sign will lead to a wrong convergent.
