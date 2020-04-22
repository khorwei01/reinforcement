# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 14:54:34 2020

@author: khorwei
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def stocksimulation(S,n,path,T,r,Sigma):
    s_space = np.zeros((path,n+1))
    deltaT = T/n
    for i in range(path):
        for j in range(n+1):
            if j == 0:
                s_space[i,j] = 0
            else:
                s_space[i,j] = s_space[i,j-1] + np.random.normal(0,1)
    return s_space*np.sqrt(deltaT) #simulted a bronian motion and ajusted by time

#call option pricing
def call(S, K, Sigma, r, T, t):
    phi=norm.cdf((np.log(S/K)+(r+1/2*Sigma**2)*(T-t))/(Sigma*np.sqrt(T-t)))
    psi=K*np.exp(-r*(T-t))*norm.cdf((np.log(S/K)+((r-1/2*Sigma**2)*(T-t)))/(Sigma*np.sqrt(T-t)))
    return phi*S-psi

#to calculate delta and respective cost
def replicate_error(S, K, Sigma, r, T, t):
    '''
    S=St
    
    '''
    delta = norm.cdf((np.log(S[:-1]/K)+(r+1/2*Sigma**2)*(T-t[:-1]))/(Sigma*np.sqrt(T-t[:-1])))
    addcost = 0
    #delta = np.zeros(len(t)-1)
    cost1 = 0 #get noth if OTM
    if S[-1]>K:
        #delta [-1] = 1
        addcost = (1-delta[-1])*S[-1]
        cost1 = K #will get K if ITM
    else:
        addcost = (0-delta[-1])*S[-1]
    cost = [S[0]*delta[0]] + [S[i]*(delta[i]-delta[i-1]) for i in range(1,len(S)-1)] 
    result = sum(cost*(np.flip(np.exp(r*t[1:])))) - cost1 + addcost
    return result

n=84
S=100
path = 50000
r = 0.05
Sigma = 0.2
T = 1/12
K = 100
smotion = stocksimulation(S,n,path,T,r,Sigma)
initial = call(S, K, Sigma, r, T, 0)
t = np.linspace(0,T, n+1)
fun = lambda x: S*np.exp((r-1/2*Sigma*Sigma)*t+Sigma*x)
check=np.zeros(path)

for i in range(path):
    St = fun(smotion[i])
    #i=0
    check[i] = replicate_error(St, K, Sigma, r, T, t) - initial
    
plt.hist(check)
plt.title('P&L of discrete delta hedge, 50000 path')
plt.xlabel('Final Profit and Loss')
plt.ylabel('Number of Trade')
plt.xlim(-3.5,3.5)

