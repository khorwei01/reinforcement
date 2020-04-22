# -*- coding: utf-8 -*-
"""
FI part 4

@author: Archer
"""
import numpy as np
import pandas as pd
from enum import Enum
from scipy.stats import norm
from scipy.integrate import quad
from scipy.misc import derivative
from math import log
from scipy.optimize import fsolve
# discounting methods
from part1 import OISdiscountFactorOutput, Fswapr
from part2 import getSABR


def sigmanN(n,N,K,m):
    N = N-n 
    S = ParSwapRate(n,N,m) 
    T = N
    alpha, rho, nu = getSABR(n, N-n)
    sigma_sabr = SABR(S, K, T, alpha, 0.9, rho, nu)
    return sigma_sabr 


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
        z = (nu/alpha)*((F*X)**(0.5*(1-beta)))*log(F/X)
        zhi = log((((1 - 2*rho*z + z*z)**0.5) + z - rho)/(1 - rho))
        numer1 = (((1 - beta)**2)/24)*((alpha*alpha)/((F*X)**(1 - beta)))
        numer2 = 0.25*rho*beta*nu*alpha/((F*X)**((1 - beta)/2))
        numer3 = ((2 - 3*rho*rho)/24)*nu*nu
        numer = alpha*(1 + (numer1 + numer2 + numer3)*T)*z
        denom1 = ((1 - beta)**2/24)*(log(F/X))**2
        denom2 = (((1 - beta)**4)/1920)*((log(F/X))**4)
        denom = ((F*X)**((1 - beta)/2))*(1 + denom1 + denom2)*zhi
        sabrsigma = numer/denom

    return sabrsigma

# import from part1 for reference
def OISref(n):
    return OISdiscountFactorOutput(n)
#OISref(3)

def ParSwapRate(n,N,m=2):
    '''underlying swap LIBOR/ collateralized
    '''
#    flt = sum([LiborRate(i-1,i)*OISref(i) for i in range(n+1,N+1,m)])
#    fix = sum([1*OISref(i) for i in range(n+1,N+1,m)])
#    return flt/fix
    return Fswapr(n,N-n)
#ParSwapRate(5,15)

class PayoffType(Enum):
    Call = 0
    Put = 1

def IRR(S,Tenor,m=2):
    '''sum of IRR discounting
    swap should pay from n+1 yr, so adjust 
    m is payment frequency
    S is par swap rate
    n,N are yrs starting swap and stop swap
    Note: 
        1. swap first payment start from n+1
        2. by default start from 1 to m*N
    '''
    comps = [1/m/(1+S/m)**i for i in range(1,Tenor*m+1)]
    return sum(comps)

def IRR_1d(S,Tenor,m=2):
    '''derivative once of IRR
    '''
    comps = [-i*(1/m**2)/(1+S/m)**(i+1) for i in range(1,Tenor*m+1)]
    return sum(comps)

def IRR_2d(S,Tenor,m=2):
    '''derivative twice of IRR
    '''
    comps = [(i*(i+1))*(1/m**3)/(1+S/m)**(i+2) for i in range(1,Tenor*m+1)]
    return sum(comps)

def Black76(S, K, r, sigma, T, PayoffType=PayoffType.Call):
    '''real Black76 should go with F=S*np.exp(r*T)
    '''
    d1 = (np.log(S/K)+(r+sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    func = {
            PayoffType.Call: lambda : S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2),
            PayoffType.Put: lambda : S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2) - S + K*np.exp(-r*T)
            }
    return func[PayoffType]()

def SwaptionPrice(Df, S0, K, swapTenor, n, PayoffType=PayoffType.Call, m=2):
    '''
    Df is Discount Factor from Tn(swap start time) to T0
    S0 refers to par swap rate observed at T0 with payment from Tn to TN
    swapTenor is swap tenor
    n is time before swap
    signanN is a function
    '''
    irr = IRR(S0, swapTenor, m)
    sigmaK = sigmanN(n=n,N=n+swapTenor,K=K,m=m)
    b76p = Black76(S0, K, 0, sigmaK, n, PayoffType=PayoffType) # take r as 0
    return Df*irr*b76p

def testSwaptionPricer():
    # swaption price test case: forward swaption 5*10
    Df = 1 # discount factor
    K = 0.1 # strike
    n=5
    swapTenor=10
    S0 = ParSwapRate(n,n+swapTenor) # par swap rate at T0
    swaptionp = SwaptionPrice(Df, S0, K, swapTenor, n, PayoffType=PayoffType.Call)
    print('Swaption price test-Call:',swaptionp)
    swaptionp = SwaptionPrice(Df, S0, K, swapTenor, n, PayoffType=PayoffType.Put)
    print('Swaption price test-Put:',swaptionp)

def CMSPrice(payoff, g_d1, g_d2, swapTenor, n, m=2):
    '''
    payoff(or g) is a function with parameter K(par swap rate)
    h_d1 is first difference with K
    h_d2 is twice difference with K
    Df from 0 to n(when swap starts)
    '''
    if (n%1 != 0) or (swapTenor%1 != 0):
        print('Do not support float numbers for now.')
        return 0
    S0 = ParSwapRate(n,n+swapTenor,m)
    Df = OISref(n)
    # specify used formulas, used analytical formulas as much as possible
    IRR_cms = lambda K: IRR(K,swapTenor)
    IRR_d1_cms = lambda K: IRR_1d(K,swapTenor)
    IRR_d2_cms = lambda K: IRR_2d(K,swapTenor)
    # h = payoff / IRR
#    h = lambda K: payoff(K)/IRR_cms(K) # not used
    h_d1 = lambda K: ( g_d1(K) / IRR_cms(K) 
                        - IRR_d1_cms(K) * payoff(K) / IRR_cms(K)**2 )
    h_d2 = lambda K: ( g_d2(K) / IRR_cms(K)
                        - IRR_d2_cms(K) * payoff(K) / (IRR_cms(K)**2)
                        - 2 * IRR_d1_cms(K) * g_d1(K) / (IRR_cms(K)**2)
                        + 2 * IRR_d1_cms(K)**2 * payoff(K)/(IRR_cms(K)**3) )
    # Df in swaption must be from Tn to T0 for swaption
    # from derivation, we know that K to swap is strike, but for payoff function, is par swap rate.
    swaption_payer = lambda K : SwaptionPrice(Df, S0, K, swapTenor, n, PayoffType=PayoffType.Call, m=m)
    swaption_receiver = lambda K : SwaptionPrice(Df, S0, K, swapTenor, n, PayoffType=PayoffType.Put, m=m)
    # within quad
    quad1 = lambda K : h_d2(K)*swaption_receiver(K)
    quad2 = lambda K : h_d2(K)*swaption_payer(K)
    # sum parts
    # p3 and p4 are intergals, which can have divergent results
    p1 = Df * payoff(S0)
    p2 = h_d1(S0)*(swaption_payer(S0)-swaption_receiver(S0))
    p3 = quad(quad1, 0, S0)[0] 
    p4 = quad(quad2, S0, np.inf)[0]
    return p1 + p2 + p3 + p4

def CMSCapletPrice(payoff, g_d1, g_d2, swapTenor, n, capletstrike, m=2):
    if (n%1 != 0) or (swapTenor%1 != 0):
        print('Do not support float numbers for now.')
        return 0
    S0 = ParSwapRate(n,n+swapTenor,m)
    Df = OISref(n)
    # specify used formulas, used analytical formulas as much as possible
    IRR_cms = lambda K: IRR(K,swapTenor )
    IRR_d1_cms = lambda K: IRR_1d(K,swapTenor )
    IRR_d2_cms = lambda K: IRR_2d(K,swapTenor )
    # h = payoff / IRR
#    h = lambda K: payoff(K)/IRR_cms(K) # not used
    h_d1 = lambda K: ( g_d1(K) / IRR_cms(K) 
                        - IRR_d1_cms(K) * payoff(K) / IRR_cms(K)**2 )
    h_d2 = lambda K: ( g_d2(K) / IRR_cms(K)
                        - IRR_d2_cms(K) * payoff(K) / (IRR_cms(K)**2)
                        - 2 * IRR_d1_cms(K) * g_d1(K) / (IRR_cms(K)**2)
                        + 2 * IRR_d1_cms(K)**2 * payoff(K)/(IRR_cms(K)**3) )

    # Df in swaption must be from Tn to T0 for swaption
    # from derivation, we know that K to swap is strike, but for payoff function, is par swap rate.
    swaption_payer = lambda K : SwaptionPrice(Df, S0, K, swapTenor, n, PayoffType=PayoffType.Call, m=m)
    swaption_receiver = lambda K : SwaptionPrice(Df, S0, K, swapTenor, n, PayoffType=PayoffType.Put, m=m)
    # within quad
    quad1 = lambda K : h_d2(K)*swaption_receiver(K) 
    quad2 = lambda K : h_d2(K)*swaption_payer(K) 
    # sum parts
    # p3 and p4 are intergals, which can have divergent results
    if capletstrike < S0: 
        p1 = Df * payoff(S0) 
        p2 = h_d1(capletstrike)*swaption_receiver(capletstrike) 
        p3 = quad(quad1, capletstrike, S0)[0] 
        p4 = quad(quad2, S0, np.inf)[0] 
    else: 
        p1 = 0
        p2 = h_d1(capletstrike)*swaption_payer(capletstrike) 
        p3 = 0
        p4 = quad(quad2, capletstrike, np.inf)[0] 
    pcheck = quad(quad1, 0, capletstrike)[0] 
    print('Caplet part2 and 3', [p2,pcheck])
    print('h1d',h_d1(capletstrike),'\nh2d',h_d2(capletstrike))
    print('rec',round(swaption_receiver(capletstrike),8))
    return p1 + p2 + p3 + p4

class backup():
    # derivative by derivative func
    IRR_d1_cms = lambda K: derivative(IRR_cms, K, dx=0.001 ,n=1)
    IRR_d2_cms = lambda K: derivative(IRR_cms, K, dx=0.001 ,n=2)
    # normal case for normal CMS paying K
    payoff = lambda K: K 
    g_d1 = lambda K: 1 
    g_d2 = lambda K: 0 
    
def peng():
    m = 2
    N = 15
    n = 5
    swapTenor = N-n
    # payoff equations
    payoff = lambda K: K 
    g_d1 = lambda K: 1 
    g_d2 = lambda K: 0 
    
#    S0 = ParSwapRate(n,n+swapTenor) 
    PV = CMSPrice(payoff, g_d1, g_d2, swapTenor, n, m)
    print('Q1: CMS PV:',PV)
    Df = OISref(n)
    print('CMS rate:',PV/Df)

def part4(n=5, N=15):
    p=4
    q=2
    swapTenor = N-n
    # payoff equations
    payoff = lambda K: K**(1/p) - 0.04**(1/q) 
    g_d1 = lambda K: (1/p)*K**(1/p-1) 
    g_d2 = lambda K: 1/p*(1/p-1)*K**(1/p-2) 
    m=2
#    S0 = ParSwapRate(n,n+swapTenor) 
    PV = CMSPrice(payoff, g_d1, g_d2, swapTenor, n, m)
    print('Q1: CMS PV:',round(PV,8))
    Df = OISref(n)
    print('CMS rate:',round(PV/Df,8))
    
    S0 = ParSwapRate(n,n+swapTenor)
    print('S0',S0)
    capletstrike = fsolve(payoff,0)[0]
    print('caplet strike', capletstrike)
    PVop = CMSCapletPrice(payoff, g_d1, g_d2, swapTenor, n, capletstrike, m)
    print('Q2: CMS Caplet PV:',round(PVop,8))
    print('difference between Option - CMS', PVop-PV)


if __name__ == '__main__':
    # so far cannot support n and N flt numbers
    part4(n=5, N=15)
    
