This project is aimed to price cms with various tenor.

1. bootstrape the OIS discount curve and Libor discount curve. The overnight OIS rate are assumed to be flat from one tenor to another tenor while the LIBOR rate are linearly interpolated.

OIS discount
<img src="https://github.com/khorwei01/reinforcement/blob/master/image/OISdf.png" width="300" height="250">

LIBOR discount
<img src="https://github.com/khorwei01/reinforcement/blob/master/image/IRSdf.png" width="300" height="250">

2. Calibration on the SABR parameter. As the problem of multiple global minimal axists, simple least sqaure function couldn't produce a consistent result. Therefore, we emply stochastic optimization algo - Basinhopping to obtain the global minimal and retrieve consistent SABR parameter. We also calibrate displaced diffusion model by employing least sqaure method.

<img src="https://github.com/khorwei01/reinforcement/blob/master/image/10by1try.png" width="300" height="250">

We also compare the pricing error betwwen calibrated price (SABR model and displaced diffussion model). Blue line: Market price - calibrated price (SABR); Yellow line: Market price - calibrated price (DD)

<img src="https://github.com/khorwei01/reinforcement/blob/master/image/totalcom.png" width="300" height="250">

3. Price & Compare forward swaption and CMS

<img src="https://github.com/khorwei01/reinforcement/blob/master/image/diff.png" width="300" height="250">


4. Price prescribe CMS payoff
