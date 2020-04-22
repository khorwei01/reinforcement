import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy import integrate
from scipy.misc import derivative
from math import sqrt,log,exp
from scipy.optimize import brentq
import matplotlib.pylab as plt
from part1 import OISdiscountFactorOutput, Fswapr
from part2 import getSABR

def D(T):
    return OISdiscountFactorOutput(T)

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


def Swaprate(n,N):
    return Fswapr(n,N)

def IRR(N,m,K,n):#N:swap duration m:payment frequency n:swap start
    irr=0
    for i in range(1,N*m+1):
        irr += (1/m)/((1+K/m)**i)
    return irr

def IRR_d(N,m,K,n):#derivative
    return derivative(lambda x:IRR(N,m,x,n),K,dx=1e-06)

def IRR_dd(N,m,K,n):#second derivative
    return derivative(lambda x:IRR(N,m,x,n),K,n=2,dx=1e-06)

def Black76_call(F,K,T,sigma):
    d1=(log(F/K)+0.5*T*sigma**2)/(sigma*sqrt(T))
    d2=d1-sigma*sqrt(T)
    c=D(T)*(F*norm.cdf(d1)-K*norm.cdf(d2))
    return c

def Black76_put(F,K,T,sigma):
    d1=(log(F/K)+0.5*T*sigma**2)/(sigma*sqrt(T))
    d2=d1-sigma*sqrt(T)
    p=D(T)*(K*norm.cdf(-d2)-F*norm.cdf(-d1))
    return p
    
def getSABR_alpharhonu(n,N):
    return getSABR(n,N) 

def V_pay(m,n,N,K,T):#IRR-settled option pricer
    S=Swaprate(n,N)
    alpha,rho,nu=getSABR_alpharhonu(n,N)
    sigma_sabr=SABR(S, K, T, alpha, 0.9, rho, nu)
    V=IRR(N,m,S,n)*Black76_call(S,K,T,sigma_sabr)
    #ignore discount factor since it can be cancelled away
    return V

def V_rec(m,n,N,K,T):#IRR-settled option pricer
    S=Swaprate(n,N)
    alpha,rho,nu=getSABR_alpharhonu(n,N)
    sigma_sabr=SABR(S, K, T, alpha, 0.9, rho, nu)
    V=IRR(N,m,S,n)*Black76_put(S,K,T,sigma_sabr)
    #ignore discount factor since it can be cancelled away
    return V

def h_dd(m,K,n,N):#second derivative of K
    part1=(-IRR_dd(N,m,K,n)*K-2*IRR_d(N,m,K,n))/IRR(N,m,K,n)**2
    part2=2*IRR_d(N,m,K,n)**2*K/IRR(N,m,K,n)**3
    return part1+part2

def CMS_rate(m,n,N,T):
    F=Swaprate(n,N)
    integrate1=integrate.quad(lambda x:\
        h_dd(m,x,n,N)*V_rec(m,n,N,x,T),F-0.02,F)[0]
    integrate2=integrate.quad(lambda x:\
        h_dd(m,x,n,N)*V_pay(m,n,N,x,T),F,F+0.02)[0]
    cms_rate=F+integrate1+integrate2
    
    return cms_rate

def PV_recleg(m,n,N):
#m:CMS payment frequency n:over next n years N: CMS N y
    PV = 0
    for i in range(1,m*n+1):
        PV += D(i/m)/m*CMS_rate(m=m,n=i/m,N=N,T=i/m)
    return PV

def part2():
    # initializaiton
    n_list=[1,5,10];N_list=[1,2,3,5,10]
    df_CMS=pd.DataFrame(index=n_list,columns=N_list)
    df_swaprate=pd.DataFrame(index=n_list,columns=N_list)
    # calculation
    for n in n_list:
        for N in N_list:
            df_CMS.loc[n,N]=round(CMS_rate(2,n,N,n),6)
            df_swaprate.loc[n,N]=round(Swaprate(n,N),6)
    # plot
    df_diff=round((df_CMS-df_swaprate),6)
    plt.xlim([0,11])
    plt.plot(N_list,df_diff.loc[n_list[0]],label='maturity=1y')
    plt.scatter(N_list,df_diff.loc[n_list[0]])
    plt.plot(N_list,df_diff.loc[n_list[1]],label='maturity=5y')
    plt.scatter(N_list,df_diff.loc[n_list[1]])
    plt.plot(N_list,df_diff.loc[n_list[2]],label='maturity=10y')
    plt.scatter(N_list,df_diff.loc[n_list[2]])
    plt.legend()
    plt.xlabel('tenor')
    plt.title('difference between forward swap rates and CMS rates')
    
#    df_swaprate.to_csv('swaprate.csv')
#    df_CMS.to_csv('CMS.csv')
#    df_diff.to_csv('diff.csv')
    return df_CMS

if __name__ == '__main__':
    #part 2
    CMS = part2()
    #part 1
    #PV1 of a leg receiveing CMS 10y semi-annually over next 5y
    PV1=PV_recleg(2,5,10)
    #PV2 of a leg receiving CMS 2y quarterly over next 10y
    PV2=PV_recleg(4,10,2)   
 