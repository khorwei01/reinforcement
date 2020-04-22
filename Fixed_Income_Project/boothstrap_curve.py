# -*- coding: utf-8 -*-
"""
part1

@author: Li Jiahang
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import fsolve


def OISclean(ois):
    # Data cleansing
    oisindex=[]
    for strT in ois['Tenor']:
        if strT[-1]=='y':
            oisindex.append(int(strT[0:-1]))
        elif strT[-1]=='m':
            oisindex.append(int(strT[0:-1])/12)
    ois.index=oisindex
    ois['OIS rate']=1.0
    ois['Discount rate']=1.0
    return ois

def Initial(T, ois):
    #OIS Tenor is completed at intervals of 1, with the complement value as the default
    if T+1 in ois.index:
        return ois
    else:
        ois.loc[T+1]=[str(int(T+1))+'y','Interp','x',1.0,1.0]
        ois = Initial(T+1, ois)
    return ois

def interpo(x,n,T,ois):
    #x denote the ois rate at T compound for 1 year
    #sum(discount)
    sumd=sum([ois['Discount rate'].loc[i] for i in range(1,int(T-n+1))])+sum([1/x**(i+1) for i in range(n-1)])*ois['Discount rate'].loc[T-n]
    y=ois['Discount rate'].loc[T-n]/(x**n)-(1-ois['Rate'].loc[T]*sumd)/(ois['Rate'].loc[T]+1)
    return y

def CalOISDiscRate(initialT,ois):
    for i in range(len(initialT)):
        T=initialT[i]
        if i==0:
            ois['Discount rate'].loc[T]=1/(T*ois['Rate'].loc[T]+1)
            ois['OIS rate'].loc[T]=360*((1/ois['Discount rate'].loc[T])**(1/180)-1)
        elif i==1:
            ois['Discount rate'].loc[T]=1/(T*ois['Rate'].loc[T]+1)
            ois['OIS rate'].loc[T]=360*((ois['Discount rate'].loc[T-0.5]/ois['Discount rate'].loc[T])**(1/180)-1)
        else:
            if T-initialT[i-1]==1:
                #F and discount are calculated by swap rate
                #sum(discount)
                sumd=sum([ois['Discount rate'].loc[i] for i in range(1,int(T))])
                ois['Discount rate'].loc[T]=(1-ois['Rate'].loc[T]*sumd)/(ois['Rate'].loc[T]+1)
                ois['OIS rate'].loc[T]=((ois['Discount rate'].loc[T-1]/ois['Discount rate'].loc[T])**(1/360)-1)*360
            else:
                #n denote step difference
                n=int(T)-int(initialT[i-1])
                x=fsolve(interpo,1,args=(n,T,ois))[0]
                for j in range(n):
                    ois.at[initialT[i-1]+j+1,'Discount rate']=ois.at[initialT[i-1],'Discount rate']/(x**(j+1))
                    ois.at[initialT[i-1]+j+1,'OIS rate']=360*(x**(1/360)-1)
    return ois

def IRS_cleansing(IRS):
    IRSindex=[]
    for strT in IRS['Tenor']:
        if strT[-1]=='y':
            IRSindex.append(int(strT[0:-1]))
        elif strT[-1]=='m':
            IRSindex.append(int(strT[0:-1])/12)
    IRS.index=IRSindex
    IRS['Forward Libor rate']=1.0
    IRS['Discount rate']=1.0
    #IRSinitialT is the initial Tenor in the supplied data
    return IRS

def IRSinitial(IRS, T):
    if T+0.5 in IRS.index:
        return IRS
    else:
        IRS.loc[T+0.5]=[str(int(T+1))+'y','Interp','x',1.0,1.0]
        IRS = IRSinitial(IRS, T+0.5)
        return IRS


#Formula 2: directly interpolate the discount rate(t<0.5 cannot be calculated)
#fois_intp=interp1d(ois.index,ois['Discount rate'],kind='cubic')

def interpoIRS(x,t,n,IRS):
    #x is the interval after the discount is evenly divided by n
    #f is forward libor
    f=[1]*n #forward libor rate
    D=[1]*n #discount rate
    
    D[0]=IRS['Discount rate'].loc[t]+x
    f[0]=(IRS['Discount rate'].loc[t]/D[0]-1)/0.5

    for i in range(1,n):
        D[i]=IRS['Discount rate'].loc[t]+x*(i+1)
        f[i]=(D[i-1]/D[i]-1)/0.5
    
    #sumf is the sum of swap rate-include(libor rate)
    sumf=0    
    for i in range(n):
        sumf=sumf+fois(t+(i+1)/2,ois)*(IRS['Rate'].loc[t+n/2]-f[i])
             
    y=sum([fois(i,ois)*(IRS['Rate'].loc[t+n/2]-IRS['Forward Libor rate'].loc[i]
    ) for i in np.arange(0.5,t+0.5,0.5)])+sumf
    #y is "PV fix-PV flt"
    return y

def Q2LIBOR(ois):
    # get data
    excel_path = 'IR Data.xlsx'
    IRS = pd.read_excel(excel_path, sheet_name='IRS')
    IRS.drop(IRS.columns[3:6],axis=1,inplace=True)
    # cleansing
    IRS = IRS_cleansing(IRS)
    IRSinitialT=IRS.index[:]
    #OIS Tenor is completed at intervals of 1, with the complement value as the default
    for T in IRSinitialT:
        if 30>T>=1:
            IRS = IRSinitial(IRS,T)
    IRS.sort_index(axis=0, ascending=True, inplace=True)
    for i in range(len(IRSinitialT)):
        #Determine whether the T computed now needs to be interpolated
        T=IRSinitialT[i]
        t=IRSinitialT[i-1]
        if T==0.5:
            IRS.at[T,'Forward Libor rate']=IRS['Rate'].loc[T]
            IRS.at[T,'Discount rate']=1/(IRS['Rate'].loc[T]*0.5+1)
        else:
            if T-t==0.5:
                #There is no need for interpolation, the data before T is complete, can be directly calculated Forward Libor rate and Discount rate
                IRS.at[T,'Forward Libor rate']=(sum([fois(i,ois)*(IRS['Rate'].loc[T]-IRS['Forward Libor rate'].loc[i]) for i in np.arange(0.5,T,0.5)])+
                                                fois(T,ois)*IRS['Rate'].loc[T])/fois(T,ois)
                IRS.at[T,'Discount rate']=IRS['Discount rate'].loc[T-0.5]/(1+0.5*IRS['Forward Libor rate'].loc[T])
            else:
                n=int((T-t)*2)
                x=fsolve(interpoIRS,0.002,args=(t,n,IRS))
                #x is the interval after the discount is evenly divided by n
                for j in range(n):
                    IRS.at[t+(j+1)/2,'Discount rate']=IRS['Discount rate'].loc[t]+x*(j+1)
                    IRS.at[t+(j+1)/2,'Forward Libor rate']=(IRS.at[t+j/2,'Discount rate']/IRS.at[t+(j+1)/2,'Discount rate']-1)/0.5
    return IRS


def Q3SWAP(Fswapr, IRS,ois):
    for t in [1,5,10]:
        for Tenor in [1,2,3,5,10]:
            print('%dy*%dy: %5.4f ' %(t,Tenor,Fswapr(t,Tenor,IRS,ois)),end=' ')
        print('\n')

def Q1OIS():
    # import data
    excel_path = 'IR Data.xlsx'
    ois = pd.read_excel(excel_path, sheet_name='OIS') 
    ois.drop(ois.columns[3:6],axis=1,inplace=True) 
    # data cleansing
    ois = OISclean(ois)
    initialT=ois.index[:]
    for T in initialT:
        if 30>T>1:
            ois = Initial(T, ois)
    ois.sort_index(axis=0, ascending=True, inplace=True)
    # cal ois rate
    ois = CalOISDiscRate(initialT,ois)
    return ois

#fit a formula
#Formula 1: calculated by the basic compound interest formula
def fois(t, ois):
    #daily convention Case by case discussion D(0,t)
    if t<=0.5:
        return 1/((1+ois['OIS rate'].loc[0.5]/360)**(t*360))
    elif 0.5<t<=1:
        return ois['Discount rate'].loc[0.5]/((1+ois['OIS rate'].loc[1]/360)**((t-0.5)*360))
    elif t>=30:
        return ois['Discount rate'].loc[30]/((1+ois['OIS rate'].loc[30]/360)**((t-30)*360))
    else:
        return ois['Discount rate'].loc[int(t)]/((1+ois['OIS rate'].loc[int(t+1)]/360)**((t-int(t))*360))

# for global call, need to put a global variable
ois = Q1OIS() 
IRS = Q2LIBOR(ois) 

foisout = lambda t: fois(t,ois)

def DLibor(T):
    if T<0.5:
        return 1-(1-IRS['Discount rate'].loc[0.5])*(T/0.5)
    else:
        dlibor=interp1d(IRS.index,IRS['Discount rate'],kind='linear')
        return dlibor(T)
def Fswapr(t,Tenor):
    return sum([((DLibor(i-0.5)/DLibor(i)-1)/0.5)*fois(i,ois) for i in np.arange(
            t+0.5,t+Tenor+0.5,0.5)])/sum([fois(i,ois) for i in np.arange(t+0.5,t+Tenor+0.5,0.5)])


#def Fswapr(t,Tenor,IRS=IRS,ois=ois):
#    return sum([IRS['Forward Libor rate'].loc[i]*fois(i,ois) for i in 
#                np.arange(t+0.5,t+Tenor+0.5,0.5)])/sum([fois(i,ois) for i in np.arange(t+0.5,t+Tenor+0.5,0.5)])
#Fswapr(5,10)

def OISdiscountFactorOutput(n):
    '''
    generic version
    '''
    if n<= 0.5:
        return 1/((1+ois.to_dict()['OIS rate'][0.5]/360)**(n*360))
#    if n == 0.5:
#        return ois.to_dict()['Discount rate'][0.5]
    yr = n//1
    m = n%1
    yrDF = ois.to_dict()['Discount rate'][yr]
    mRfed = ois.to_dict()['OIS rate'][yr]
    mDF = (1+mRfed/360)**(-360*m)
    return yrDF*mDF
#OISdiscountFactorOutput(3.2)

def fwdLiborRate(n,N):
    '''need revision
    for now, return a dictionary, with all avilable tenors and discount factors
    reutrn:
        dict: key -> tenor, value -> discount factor
    '''
    Df = lambda x : np.interp(x, IRS.index.to_list(),IRS['Discount rate'].to_list())
    LibPay = Df(n)/Df(N)-1
    return LibPay/(N-n)
#fwdLiborRate(3,4)

if __name__ == '__main__':
    ois = Q1OIS() 
    # visualize
    plt.plot(ois.index.values,ois['Discount rate'].values,label='Discount rate of OIS')
    plt.xlabel('time to maturity')
    plt.ylabel('discount rate of OIS')
    plt.legend()
    plt.show()

    IRS = Q2LIBOR(ois) 
    # visualize
    plt.plot(IRS.index.values,IRS['Forward Libor rate'].values,label='Forward Libor rate')
    plt.xlabel('time to maturity')
    plt.ylabel('Forward Libor rate')
    plt.legend()
    plt.show()

    
    Q3SWAP(Fswapr, IRS,ois) 

    

