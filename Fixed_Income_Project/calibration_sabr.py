# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 00:20:21 2020

@author: khorwei
"""

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
import math
from scipy.stats import norm
from scipy.optimize import basinhopping
from part1 import foisout as fois_1
from part1 import Fswapr as Fswapr1


def fois(e):
    return fois_1(e)

def Fswapr(t,tenor):
    return Fswapr1(t,tenor)

class MyBounds(object):
     def __init__(self, xmax=[0.4,0.4,0.4], xmin=[0.01,-0.5,0.01] ):
         self.xmax = np.array(xmax)
         self.xmin = np.array(xmin)
     def __call__(self, **kwargs):
         x = kwargs["x_new"]
         tmax = bool(np.all(x <= self.xmax))
         tmin = bool(np.all(x >= self.xmin))
         return tmax and tmin

def blackSwaptionPricer(S, K, sigma, t, df, trade = None):
    d1 = (math.log(S/K) + sigma *sigma* t /2) / sigma / math.sqrt(t)
    d2 = d1 - sigma * math.sqrt(t)
    if trade == None:
        if S > K:
            return df * (K * norm.cdf(-d2) - S* norm.cdf(-d1))
        else:
            return df * (S* norm.cdf(d1) - K * norm.cdf(d2))
    if trade ==  'Payer':
        price = df * (S* norm.cdf(d1) - K * norm.cdf(d2))
        return price
    else:
        price = df * (K * norm.cdf(-d2) - S* norm.cdf(-d1))
        return price

def ddSwaptionPricer(S, K, sigma, t, B, df, trade = None):
    sigmaPrime = sigma * B
    sPrime = S / B
    kPrime = K+ (1-B)/B*S
    xPrime = (math.log(round(kPrime / sPrime, 5)) + sigmaPrime**2*t /2) / sigmaPrime / math.sqrt(t)
    if trade == None:
        if S > K:
            return df * (kPrime * norm.cdf(xPrime) - sPrime * norm.cdf(xPrime-B*sigma*math.sqrt(t)))
        else:
            return df * (sPrime * norm.cdf(-xPrime+B*sigma*math.sqrt(t)) - kPrime * norm.cdf(-xPrime))
    if trade == 'Payer':       
        return df * (sPrime * norm.cdf(-xPrime+B*sigma*math.sqrt(t)) - kPrime * norm.cdf(-xPrime))
    else:
        return df * (kPrime * norm.cdf(xPrime) - sPrime * norm.cdf(xPrime-B*sigma*math.sqrt(t)))
    
def SABR(F, K, T, alpha, beta, rho, nu):
    X = K
    # if K is at-the-money-forward
    if abs(F - K) < 1e-12:
        numer1 = (((1 - beta)**2)/24)*alpha*alpha/(F**(2 - 2*beta))
        numer2 = 0.25*rho*beta*nu*alpha/(F**(1 - beta))
        numer3 = ((2 - 3*rho*rho)/24)*nu*nu
        VolAtm = alpha*(1 + (numer1 + numer2 + numer3)*T)/(F**(1-beta))
        sabrsigma = VolAtm
    else:
        z = (nu/alpha)*((F*X)**(0.5*(1-beta)))*np.log(F/X)
        zhi = np.log((((1 - 2*rho*z + z*z)**0.5) + z - rho)/(1 - rho))
        numer1 = (((1 - beta)**2)/24)*((alpha*alpha)/((F*X)**(1 - beta)))
        numer2 = 0.25*rho*beta*nu*alpha/((F*X)**((1 - beta)/2))
        numer3 = ((2 - 3*rho*rho)/24)*nu*nu
        numer = alpha*(1 + (numer1 + numer2 + numer3)*T)*z
        denom1 = ((1 - beta)**2/24)*(np.log(F/X))**2
        denom2 = (((1 - beta)**4)/1920)*((np.log(F/X))**4)
        denom = ((F*X)**((1 - beta)/2))*(1 + denom1 + denom2)*zhi
        sabrsigma = numer/denom
    return sabrsigma

def sabrcalibration(x, strikes, vols, F, T):
    err = 0.0
    punish = [5,1,1,1,1,100,1,1,1,1,5]
    for i, vol in enumerate(vols):
        err += (vol - SABR(F, strikes[i], T,
                           x[0], 0.9, x[1], x[2]))**2*punish[i]
    return err

def ddcalibration(x, strikes, price, df, T, S):
    err = 0.0
    punish = [70,1,1,1,1,100,1,1,1,1,50]
    for i, price in enumerate(price):
        err += (price*10000 - ddSwaptionPricer(S, strikes[i], x[0], T, x[1], df)*10000)**2*punish[i]
    return err
i=10
e=10
t=1
S=0.0421895275415083
def sabrParameter(swaprate):
    excel_path = 'IR Data.xlsx'
    marketvol = pd.read_excel(excel_path, sheet_name='Swaption', header=2).values
    xmin, xmax = [0.01,-0.99,0.01], [0.99,0.99,0.99]
    bounds = [(low, high) for low, high in zip(xmin, xmax)]
    minimizer_kwargs = dict(method='L-BFGS-B', bounds = bounds)
    mybounds = MyBounds()
    sabralpha = dict()
    sabrrho = dict()
    sabrnu = dict()
    for i in range(len(swaprate)):
        e = swaprate[i][0][0] #option ex[iry]
        t = swaprate[i][0][1] #swap tenor 
        S = swaprate[i][1] #swap rate
        con = [-200, -150, -100, -50, -25, 0, 25, 50, 100, 150, 200] # convention
        sig_all = [j for j in marketvol[i,2:]/100] # to get ATM sig
        strike = [S + i/10000 for i in con] # for all sigma
        F=S
        initialGuess = [0.3, -0.2, 0.1]
        res = basinhopping(lambda x: sabrcalibration(x, strike, sig_all, F, e), initialGuess,minimizer_kwargs=minimizer_kwargs,niter=100, accept_test=mybounds)
        sabralpha[swaprate[i][0]] = res.x[0]
        sabrrho[swaprate[i][0]] = res.x[1]
        sabrnu[swaprate[i][0]] = res.x[2]
    return pd.DataFrame(list(zip(sabralpha.values(),sabrrho.values(),sabrnu.values())), index = sabralpha.keys())

def ddParameter(swaprate):
    excel_path = 'IR Data.xlsx'
    marketvol = pd.read_excel(excel_path, sheet_name='Swaption', header=2).values
    ddsigma=dict()
    ddbeta=dict()
    for i in range(len(swaprate)):
        e = swaprate[i][0][0] #option ex[iry]
        t = swaprate[i][0][1] #swap tenor 
        df = fois(e)
        S = swaprate[i][1] #swap rate
        con = [-200, -150, -100, -50, -25, 0, 25, 50, 100, 150, 200]
        sig_all = [j for j in marketvol[i,2:]/100]
        strike = [S + i/10000 for i in con]  
        test1 = 10000000
        sss = 0
        bbb = 0
        price = [blackSwaptionPricer(S, strike[i], sig_all[i], e, df) for i in range(len(strike))]
        for j in range(1,10):
            for k in range(1,10):
                initialGuess = [j*10/100, k*10/100]
                bounds=((0.01,0.01),(0.99,0.99))
                res = least_squares(lambda x: ddcalibration(x, strike, price, df, e, S),initialGuess, bounds=bounds)
                test = res.cost/10000
                if test < test1:
                    test1 = test
                    sss = res.x[0]
                    bbb = res.x[1]
        ddsigma[swaprate[i][0]] = sss
        ddbeta[swaprate[i][0]] = bbb
    return pd.DataFrame(list(zip(ddsigma.values(),ddbeta.values())), index = ddsigma.keys())

def getDD(inTime, forTime, ddPara):
    loc = None
    for i in range(5):
        if forTime == ddPara.iloc[i,1]:
            loc = i
    if loc == None:
        return print('No Swap tenor match')
    if inTime < 1:
        return ddPara.iloc[loc,2], ddPara.iloc[loc,3]
    elif inTime > 10 :
        return ddPara.iloc[10+loc,2], ddPara.iloc[10+loc,3]
    else:
        funsi = interp1d([1,5,10], ddPara.iloc[[loc,loc+5,loc+10],2].values)
        funbe = interp1d([1,5,10], ddPara.iloc[[loc,loc+5,loc+10],3].values)
        newsi = float(funsi(inTime))
        newbe = float(funbe(inTime))
        return newsi, newbe
    
def vSwaption(S, K, df, dayfraction, inTime, forTime, sabrPara=None, ddPara=None, payoff = None):
    n = 1/dayfraction
    if ddPara == None:
        sig = SABR(S, K, inTime, alpha = sabrPara[0], beta = 0.9, rho = sabrPara[1], nu = sabrPara[2])
        return blackSwaptionPricer(S, K, sig, inTime, df, trade = payoff)
    else:
        return ddSwaptionPricer(S, K, ddPara[0], inTime,ddPara[1], df, trade = payoff)

def Q2prep():
    excel_path = 'IR Data.xlsx'
    inYear = [1,5,10]
    forYear = [1,2,3,5,10]
    fsrTenor = []
    swap_rate = []
    for i in inYear:
        for j in forYear:
            fsrTenor.append((i,j))
            swap_rate.append(Fswapr(i,j))
    return pd.DataFrame(list(zip(fsrTenor,swap_rate)),columns=['tenor', 'swap_rate'])   

def Q2sabrcalibresult():
    excel_path = 'sabrdict.csv'
    sabrdata = pd.read_csv(excel_path, names= ['Expiry','Tenor', 'Alpha','Rho','Nu'], header = 0)
    return print(sabrdata)

def Q2DDcalibresult():
    excel_path = 'ddict.csv'
    dddata = pd.read_csv(excel_path, names= ['Expiry','Tenor', 'Sigma','Beta'], header = 0)
    return print(dddata)

sabrdata = pd.read_csv('sabrdict.csv', names= ['Expiry','Tenor', 'Alpha','Rho','Nu'], header = 0)
dddata = pd.read_csv('ddict.csv', names= ['Expiry','Tenor', 'Sigma','Beta'], header = 0)
def getSABR(inTime, forTime, sabrPara= sabrdata):
    '''
    for sabr result reuse
    '''
    loc = None
    for i in range(5):
        if forTime == sabrPara.iloc[i,1]:
            loc = i
            break
    if loc == None:
        return print('No Swap tenor match')
    if inTime < 1:
        return sabrPara.iloc[loc,2], sabrPara.iloc[loc,3], sabrPara.iloc[loc,4]
    elif inTime > 10 :
        return sabrPara.iloc[10+loc,2], sabrPara.iloc[10+loc,3], sabrPara.iloc[10+loc,4]
    else:
        funal = interp1d([1,5,10], sabrPara.iloc[[loc,loc+5,loc+10],2].values)
        funrh = interp1d([1,5,10], sabrPara.iloc[[loc,loc+5,loc+10],3].values)
        funnu = interp1d([1,5,10], sabrPara.iloc[[loc,loc+5,loc+10],4].values)
        newal = float(funal(inTime))
        newrh = float(funrh(inTime))
        newnu = float(funnu(inTime))
        return newal, newrh, newnu
    
def Q2part2Rec():
    part2 = [Fswapr(8,10)]
    df = fois(8)
    rec8by10dd = [round(vSwaption(part2[0], i, df, 0.5, 8, 10, sabrPara=None, ddPara=getDD(8, 10, dddata), payoff = 'Receiver'),6)for i in [0.01*j for j in range(1,9)]]
    rec2by10sabr = [round(vSwaption(part2[0], i, df, 0.5, 8, 10, sabrPara=getSABR(8, 10, sabrPara = sabrdata), ddPara=None, payoff = 'Receiver'),6)for i in [0.01*j for j in range(1,9)]]
    result1 = pd.DataFrame([rec8by10dd, rec2by10sabr], index = ['DD','SABR'],columns = ['Strike:' + str(0.01*i) for i in range(1,9)])                    
    return print(result1)  

def Q2part2Pay():
    part2 = [Fswapr(2,10)]
    df = fois(2)
    pay2by10dd = [round(vSwaption(part2[0], i, df, 0.5, 2, 10, sabrPara=None, ddPara=getDD(2, 10, dddata), payoff = 'Payer'),6) for i in [0.01*j for j in range(1,9)]]
    pay2by10sabr = [round(vSwaption(part2[0], i, df, 0.5, 2, 10, sabrPara=getSABR(2, 10, sabrPara = sabrdata), ddPara=None, payoff = 'Payer'),6) for i in [0.01*j for j in range(1,9)]]
    result = pd.DataFrame([pay2by10dd, pay2by10sabr], index = ['DD','SABR'],columns = ['Strike:' + str(0.01*i) for i in range(1,9)])                    
    return print(result) 



if __name__ == '__main__':
    
#    Choose = input('Warning: running the whole code will spend quit some time, do you want to rerun the results?(Y/N)')
#    if Choose == 'Y':        
#        #SABR_calibrated_data = sabrParameter(swaprate = Q2prep().values)
#        #Displaced_Diffusion_calibrated data = ddParameter(swaprate = Q2prep().values)
#        print('Please wait, starting running the code.')
#        sabrdict = sabrParameter(swaprate = Q2prep().values)
#        ddict = ddParameter(swaprate = Q2prep().values)
#        sabrdict.to_csv(r'sabrdict.csv')
#        ddict.to_csv(r'ddict.csv')   
#    #command to import
#    #Q2sabrcalibresult()
#    #Q2DDcalibresult()
#    
#    #Calibration result import directly to avoid running too long
#    sabrdata = pd.read_csv('sabrdict.csv', names= ['Expiry','Tenor', 'Alpha','Rho','Nu'], header = 0)
#    dddata = pd.read_csv('ddict.csv', names= ['Expiry','Tenor', 'Sigma','Beta'], header = 0)
#    print(sabrdata)
#    print(dddata)
    
#    Q2part2Pay()
#    Q2part2Rec()
    
    
    # sample usage of quoting sabr
    for n in [0.25,2.125]:
        for tenor in [5,10]:
            alpha, beta, rho = getSABR(inTime=n, forTime=tenor)
            print(n,tenor,[alpha,beta,rho])
