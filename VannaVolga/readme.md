## FX market volatility smile.  
Vanna Volga is a well-known method in FX market for volatility calibration, espxpecially in priicing touch / no touch-like exotic option. Aim of such technique is to price in the Vanna and Volga cost of option for hedging purpose.  

To best price an FX option accurate and fast way, we compare the result of Vanna Volga method, quadratic itnerpolation, 1st approximationa and 2nd approximation as below and want the best performer.

<img src="https://github.com/khorwei01/reinforcement/blob/master/image/ON_USDJPY_VV_compare.png" width="350" height="250">

For most of the time, we may want to quote option from delta 25 to delta 75 only. Then we compare the Vol difference between VV method and Quadratic iinterpolation in 3 dimension

<img src="https://github.com/khorwei01/reinforcement/blob/master/image/Diff_VV_Quad_3D.png" width="350" height="250">

Lastly, show quadratic interpolated voolatilit in 3D.

<img src="https://github.com/khorwei01/reinforcement/blob/master/image/quadratic_FX_3D.png" width="350" height="250">
