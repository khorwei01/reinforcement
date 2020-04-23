This work attempted to replicate the idea of Derman, to use the market option quote (across maturity and strike) to backup a implied tree.

According to the paper example, we have the graph below.

Note: some problem from the idea and potential solution

1. have tried to match the actual market option, the pricer failed in longer maturity (i.e. more than 0.5 y)** want to try trinomial tree for this

2. adjust the log location of the price at the wings according to the paper as the wing is not spanning as longer tenor. However, the log location adjustment is helping a bit.

3. the time step cannot to be too many(optimal is about 50 step of tree), the limit movement of wing disorted the pricer.

<img src="https://github.com/khorwei01/reinforcement/blob/master/image/implied.png" width="350" height="250">
