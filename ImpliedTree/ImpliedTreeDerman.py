# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 22:58:10 2020

@author: khorwei
"""
from scipy.stats import norm
import math
import matplotlib.pyplot as plt
import itertools
from enum import Enum
import numpy as np

class PayoffType(Enum):
    Call = 0
    Put = 1


def crrEuroBionomial(S, r, vol, PayoffType, K, tau, n):
    t = tau
    u = math.exp(vol)
    d = math.exp(-vol)
    p = (math.exp(r * t) - d) / (u-d)
    payoffDict = {PayoffType.Call:lambda s:max(s-K,0) , PayoffType.Put:lambda s:max(K-s,0)}
    #Last slice of the node
    vs = [payoffDict[PayoffType](S * u**(n-i-i)) for i in range(n+1)]
    for i in range(n-1,-1,-1):
        for j in range(i+1):
            vs[j] = math.exp(-r*t)*(p * vs[j] + (1-p) * vs[j+1])
    return vs[0]

def forward(nodeS, r, tau):
    return nodeS * math.exp(r*tau)

def bigSigma_call(nodeS, lambd, i, j, r, tau):
    if j == 0:
        return 0
    return sum([lambd[i - 1][k] * (forward(nodeS[i - 1][k], r, tau) - nodeS[i - 1][j]) for k in range(0,j)])


def bigSigma_put(nodeS, lambd, i, j, r, tau):
    if j-1 == i:
        return 0
    return sum([lambd[i - 1][k] * (nodeS[i - 1][j] - forward(nodeS[i - 1][k], r, tau) ) for k in range(j+1, i)])

def option_BS_call(S,K,T,sigma,r,q):
    d1=(np.log(S/K)+(r-q+sigma**2/2)*T)\
       /(sigma*np.sqrt(T))
    d2=d1-sigma*np.sqrt(T)
    c=S*np.exp(-q*T)*norm.cdf(d1)- \
      K*np.exp(-r*T)*norm.cdf(d2)
    return c

def option_BS_put(S,K,T,sigma,r,q):
    d1=(np.log(S/K)+(r-q+sigma**2/2)*T)\
       /(sigma*np.sqrt(T))
    d2=d1-sigma*np.sqrt(T)
    p=K*np.exp(-r*T)*norm.cdf(-d2)- \
      S*np.exp(-q*T)*norm.cdf(-d1)
    return p

stock = [[110.52, 90.48],[120.27, 100, 79.30],[130.09, 110.60, 90.42, 71.39],[139.78, 120.51, 100, 79.43, 71.39],[147.52, 130.15, 110.61, 90.41, 71.27, 54.48]]
siigma = [[0],[0.1], [0.0947, 0.1047]]
siigma = [[0],[0.1]]+[[0.1 + ((100-stock[j][i])/100)/10*0.5 for i in range(len(stock[j])) ] for j in range(len(stock))]

def callAbove(nodeS, r, i, j, tau,K = None):
    S = nodeS[0][0]
    if K == None:
        K = nodeS[i-1][j]
    else:
        K = K
    T =  tau #* i
 #   iv = createTestImpliedVol(S, r, 0)
    sigma = siigma[i][j]# iv.Vol(T, K)
    q = 0
    return crrEuroBionomial(S=S, r=r, vol=sigma, PayoffType = PayoffType.Call, K = K, tau=T, n=i)#option_BS_call(S,K,T,sigma,r,q)

def putBelow(nodeS, r, i, j, tau, K = None):
    S = nodeS[0][0]
    if K == None:
        K = nodeS[i-1][j]
    else:
        K = K
    t = tau #* 1
    sigma = siigma[i][j] #iv.Vol(T, K)
    q = 0
    return crrEuroBionomial(S=S, r=r, vol=sigma, PayoffType = PayoffType.Put, K = K, tau =t, n=i)

def nodeSforEvenUp(nodeS, lambd, i, j, r, tau):
    upper = nodeS[i][j+1] * (math.exp(r*tau) * callAbove(nodeS, r, i, j, tau) - bigSigma_call(nodeS, lambd, i, j, r, tau)) - lambd[i-1][j] * nodeS[i-1][j] * (forward(nodeS[i-1][j],r ,tau) - nodeS[i][j+1])
    lower = math.exp(r*tau) * callAbove(nodeS, r, i, j, tau) - bigSigma_call(nodeS, lambd, i, j ,r, tau) - lambd[i-1][j] * (forward(nodeS[i-1][j],r ,tau) - nodeS[i][j+1])
    return upper/lower

i=2
j=1
def nodeSforEvenDown(nodeS, lambd, i, j, r, tau):
    upper = nodeS[i][j] * (math.exp(r*tau) * putBelow(nodeS, r, i, j, tau) - bigSigma_put(nodeS, lambd, i, j, r, tau)) + lambd[i-1][j] * nodeS[i-1][j] * (forward(nodeS[i-1][j],r ,tau) - nodeS[i][j])
    lower = math.exp(r*tau) * putBelow(nodeS, r, i, j, tau) - bigSigma_put(nodeS, lambd, i, j, r, tau) + lambd[i-1][j] * (forward(nodeS[i-1][j],r ,tau) - nodeS[i][j])
    return upper/lower

def nodeForOdd(nodeS, lambd, r, tau, i):
    j = math.ceil(i/2)-1
    upper = nodeS[0][0] * (math.exp(r * tau) * callAbove(nodeS, r, i, j, tau, nodeS[0][0]) + lambd[i-1][j] * nodeS[0][0] - bigSigma_call(nodeS, lambd, i, j, r, tau))
    lower = lambd[i-1][j] * forward(nodeS[i-1][j], r, tau) - math.exp(r * tau) * callAbove(nodeS, r, i, j, tau, nodeS[0][0]) + bigSigma_call(nodeS, lambd, i, j, r, tau)
    return upper / lower 


n = 5 # how mny steps
lambd  = [[1]*i for i in range(1, n+1+1)] #collection of Arrow-Debreu prices
p = [[1]*i for i in range(1, n+1+1)] #collection of risk neutral p
#nodeS = [[1.25805]*i for i in range(1, n+1+1)] # collection of Stock price
nodeS = [[100]*i for i in range(1, n+1+1)] # collection of Stock price
r = 0.03
BigT =5
#def impliedTree(nodeS, lambd, p, r, T, n):
tau = BigT/n
for i in range(1, n+1):
    if i % 2 == 0:
        upper, lower = int(i/2), int(i/2)
        nodeS[i][upper] = nodeS[0][0]
    else:
        upper = math.ceil(i/2)-1
        lower = math.ceil(i/2)

        nodeS[i][upper] = nodeForOdd(nodeS, lambd, r, tau, i)
        nodeS[i][lower] = nodeS[0][0] ** 2 / nodeS[i][upper]
        
    if i > 1 :
        for k in range(upper -1, -1, -1) :
            nodeS[i][k] = nodeSforEvenUp(nodeS, lambd, i, k, r, tau)
        for k in range(lower, i , 1):
            nodeS[i][k+1] = nodeSforEvenDown(nodeS, lambd, i, k, r, tau)

    for j in range(i):
        p[i-1][j] = (nodeS[i - 1][j] * math.exp(r * tau) - nodeS[i][j + 1])/(nodeS[i][j] - nodeS[i][j + 1])

        if j == 0:
            lambd[i][j] = math.exp(-r * tau) * p[i-1][j] * lambd[i-1][j]

        else:
            q = 1-p[i-1][j-1]
            lambd[i][j] = math.exp(-r * tau) * q * lambd[i-1][j-1] + p[i-1][j] * lambd[i-1][j]
                
        if j == i-1:
            q = 1-p[i-1][j]
            lambd[i][j+1] = math.exp(-r * tau) * q * lambd[i-1][j]
        
        print((lambd[i][j],i,j))
x = list(itertools.chain(*nodeS))
y = list(itertools.chain(*[[1*n]*n for n in range(1,n+2)]))
for i in range(int(len(y)-max(y))):
    t , s = y[i],x[i]
    for j in range(2):
        t1 = y[i+t+j]
        s1 = x[i+t+j]
        plt.plot([t,t1],[s,s1], 'bo--')
        plt.title('Implied Option Tree')
        plt.xlabel('Time step')
        plt.ylabel('Price')
        plt.xlim(0,7)
        plt.ylim(40,160)

