This simulation aim to practise discrete delta hedging.

Suppose we have an option sold, and we shall hedge it through underlying asset. To practise continuous hedging is impossible in reality but we can increase the number of hedge within the tenor.

Of course we are not including interest rate and commission here.

Hedge 21 times in 21 trading days

<img src="https://github.com/khorwei01/reinforcement/blob/master/image/21trial.png" width="350" height="250">

mean = 0.009
standard dev(%) = 0.004%

Hedge 84 times in 21 trading days

<img src="https://github.com/khorwei01/reinforcement/blob/master/image/21trial.png" width="350" height="250">

mean = 0.009
standard dev(%) = 0.002%

Standard deviation decrease by <img src="https://render.githubusercontent.com/render/math?math=\frac{\sqrt(84)}{sqrt(21)}">
