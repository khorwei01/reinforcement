#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 16:07:17 2021

@author: khorweisheng
"""
import numpy as np
import scipy.optimize as so
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
import json
import scipy.interpolate as si
import plotly.graph_objects as go
from mpl_toolkits import mplot3d

def erf(x):
    # save the sign of x
    #sign = np.array([1 if k >= 0 else -1 for k in x])
    if len(x) >1:
        sign = np.where(x<0,-1,1)
    else:
        sign = 1 if x >= 0 else -1
    x = np.absolute(x)

    # constants
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911

    # A&S formula 7.1.26
    t = 1.0/(1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*np.exp(-x*x)
    return sign*y # erf(-x) = -erf(x)

def norm_cdf(x):
    return 0.5 * ( 1 + erf(x / np.sqrt(2)))

def norm_pdf(x):
    return np.exp(-0.5 * x * x) / (2 * np.pi) 

def european_option(S0,K,r,d, sigma, t, typ = 'C'):
    d1 = (np.log(S0 / K) + ((r-d)+ 0.5 * sigma * sigma) * t) / (sigma * np.sqrt(t))
    d2= d1 - sigma * np.sqrt(t)
    
    if typ == 'C':
        return np.exp(-r*t) * (S0 * np.exp((r-d)*t) * norm_cdf(d1) - K * norm_cdf(d2))
    elif typ == 'P':
        return np.exp(-r*t) * ( K * norm_cdf(-d2) - S0 * np.exp((r-d)*t) * norm_cdf(-d1))
    elif typ == 'Binary_C':
        return np.exp(-r*t) * norm_cdf(d2)
    elif typ == 'Binary_P':
        return np.exp(-r*t) * norm_cdf(-d2)
    else:
        raise Exception("Please specify correct type of European option (C or P), %s is not supported." %typ)


# Vanilla option delta  
def delta(S0,K,r,d, sigma, t, typ = 'C'):
    d1 = (np.log(S0 / K) + ((r-d)+ 0.5 * sigma * sigma) * t) / (sigma * np.sqrt(t))
    if typ =='C':
        return np.exp(-d*t) * norm_cdf(d1)
    elif typ =='P':
        return -np.exp(-d*t) * norm_cdf(-d1)
    else: 
        raise Exception("Please specify correct type of European option (C or P), %s is not supported." %typ)

# Vanilla option Vega
def vega(S0, K, r, d, t, sigma):
    d1_k = (np.log(S0 / K) + ((r-d)+ 0.5 * sigma * sigma) * t) / (sigma * np.sqrt(t))
    return S0 * np.exp((-d)*t)*np.sqrt(t)*norm_pdf(d1_k)

# Weigth of K to zeroised Vanna and Volga
def x3_k(S0, r, d, t, sigma_bs, k, k1, k2, k3):
    loc = np.array([k1==k, k2==k, k3==k])
    if any(loc):
        return loc * 1
    else:
        v = vega(S0, k, r, d, t, sigma_bs)
        v1 = vega(S0, k1, r, d, t, sigma_bs)
        v2 = vega(S0, k2, r, d, t, sigma_bs)
        v3 = vega(S0, k3, r, d, t, sigma_bs)
        x1 = v / v1 * np.log(k2 / k) * np.log(k3 / k) / (np.log(k2 / k1) * np.log(k3 / k1))
        x2 = v / v2 * np.log(k / k1) * np.log(k3 / k) / (np.log(k2 / k1) * np.log(k3 / k2))
        x3 = v / v3 * np.log(k / k1) * np.log(k / k2) / (np.log(k3 / k1) * np.log(k3 / k2))
        return np.array([x1, x2, x3])

def k_finder(c_25, c_50, c_75, t, r, d, S0):
    
    k3 = so.fsolve(lambda x : delta(S0, x, r, d, c_25, t, typ = 'C') - 0.25, x0 = S0)
    
    k1 = so.fsolve(lambda x: delta(S0, x, r, d, c_75, t, typ = 'P') + 0.25, x0 = S0)
    
    k2= so.fsolve(lambda x: delta(S0, x, r, d, c_50, t, typ = 'C') - 0.50, x0 = S0)
    
    return k1, k2, k3

def x_giver(k, k1, k2, k3):
    xx1 = np.log(k2 / k) * np.log(k3 / k) / (np.log(k2 / k1) * np.log(k3 / k1))
    xx2 = np.log(k / k1) * np.log(k3 / k) / (np.log(k2 / k1) * np.log(k3 / k2))
    xx3 = np.log(k / k1) * np.log(k / k2) / (np.log(k3 / k1) * np.log(k3 / k2))
    return xx1, xx2, xx3

def first_approx(S0, r, d, t, k, sigma_1, sigma_2, sigma_3):
    k1, k2, k3 = k_finder(c_25, c_50, c_75, t, r, d, S0)
    xx1, xx2, xx3 = x_giver(k, k1, k2, k3)
    
    return xx1 * sigma_1 + xx2 * sigma_2 + xx3 * sigma_3

def second_approx(S0,k, sigma_1, sigma_2, sigma_3, r, d, t):
    
    d1 = lambda x: (np.log(S0 / x) + ((r-d)+ 0.5 * sigma_2 * sigma_2) * t) / (sigma_2 * np.sqrt(t))
    d2 = lambda x: d1(x) - sigma_2 * np.sqrt(t)
    
    k1, k2, k3 = k_finder(c_25, c_50, c_75, t, r, d, S0)
    xx1, xx2, xx3 = x_giver(k, k1, k2, k3)
    
    big_d1 = xx1*sigma_1 + xx2 * sigma_2 + xx3 * sigma_3 - sigma_2
    big_d2 = xx1 * d1(k1) * d2(k1) * (sigma_1 - sigma_2)**2 + xx3 * d1(k3) * d2(k3) * (sigma_3  - sigma_2)**2
    root_term = np.sqrt(sigma_2*sigma_2 + d1(k) * d2(k) * (2 * sigma_2 * big_d1 + big_d2))    
    
    return sigma_2 + (-sigma_2 + root_term) / (d1(k) * d2(k))


## Trial 1
def massage_feed(address):
    df = pd.read_csv(address, index_col =0)
    df.ts = pd.to_datetime(df.ts, format='%Y-%m-%dT%H:%M:%S')
    df['hours'] = df.ts.dt.hour
    #df = df.loc[df.ts.dt.day < 30,:]
    df.sort_values('ts', inplace=True)
    return df

def sigma_vv(S0, k, r, d, sigma_atm, c_25, c_75, t):
    
    k1, k2, k3  = k_finder(c_25, sigma_atm, c_75, t, r, d, S0)
    
    x_atm = european_option(S0,k,r,d, sigma_atm, t, typ = 'C')
    c_m1 =  european_option(S0,k1,r,d, c_75, t, typ = 'C')
    c_m2 =  european_option(S0,k2,r,d, sigma_atm, t, typ = 'C')
    c_m3 =  european_option(S0,k3,r,d, c_25, t, typ = 'C')
    c_bs1 =  european_option(S0,k1,r,d, sigma_atm, t, typ = 'C')
    c_bs2 =  european_option(S0,k2,r,d, sigma_atm, t, typ = 'C')
    c_bs3 =  european_option(S0,k3,r,d, sigma_atm, t, typ = 'C')
    call_k_i = x_atm + x3_k(S0, r, d, t, sigma_atm, k, k1, k2, k3).transpose() @ (np.array([c_m1 - c_bs1, c_m2 - c_bs2, c_m3 - c_bs3]))
    res = so.fsolve( lambda x: abs(call_k_i[:,0] - european_option(S0, k, r, d, x, t, typ = 'C')), x0 = sigma_atm)
    
    return res 

def tem_3d_surface(t, full_3d):
    full = np.array(full_3d).reshape(-1,3)
    t_1 = np.sort(np.unique(full[:,0]))
    
    if (t>max(t_1) or t< min(t_1)):
        raise  Exception("Time interpolation failed due to t chosen fall out of bound. Choose t within %s and %s"%min(t_1) %max(t_1))
    
    
    elif t in t_1:
        
        linear = full[full[:,0]==t,1:]

        return linear
    
    else:   
        #print('nono')
        loc = np.argwhere(t_1>t)[0,0]
        idx_loc = t_1[loc-1:loc+1]
        sig_1 = full[full[:,0] == idx_loc[0], 1:] 
        sig_2 = full[full[:,0] == idx_loc[1], 1:]
        w = (idx_loc[1] - t) / (idx_loc[1] - idx_loc[0])
        linear = w * sig_1 + (1 - w) *sig_2
        
        return linear

## USDJPY 3 Aug 2021
S0 = 109.308
r = 0
d = 0

coll = {'1': [0.0643, 0.0648, 0.0676],
 '7': [0.0530125, 0.05365, 0.0565375],
 '14': [0.05233751, 0.05315001, 0.05631251],
 '21': [0.049775, 0.05065, 0.053575],
 '31': [0.0548375, 0.05585, 0.0593125],
 '59': [0.05485, 0.05585, 0.05985],
 '92': [0.054875, 0.055975, 0.060525],
 '184': [0.0574875, 0.058575, 0.0640125],
 '269': [0.0590125, 0.060275, 0.0662875],
 '365': [0.0605125, 0.061725, 0.0682875]}
list_3d = []
list1st_3d = []
list2nd_3d = []
list_quad = []
delta_3d = np.linspace(0.01,0.99,99)
duration = [int(i) for i in coll.keys()]
# for loop duration
for j in range(duration):
    t = duration[j] / 365
    
    # Geting respective market vol
    #vol = vol_dict['surface'][str(duration[j])]['smile']
    c_25 = coll[str(duration[j])][0]#vol['25']
    c_50 = coll[str(duration[j])][1]#vol['50']
    c_75 = coll[str(duration[j])][2]#vol['75']
    #coll[str(duration[j])] = [vol['25'],vol['50'],vol['75']]
    sigma_atm = c_50
    
    # Quadratic interpolation
    fun = np.polyfit([0.75, 0.5, 0.25], [c_75, c_50, c_25], 2)
    inpt = np.linspace(0.01, 0.99, 99)
    quad_fun = np.poly1d(fun)
    quad = quad_fun(inpt)
    list_quad.append(np.stack(([t,]*len(inpt), inpt, quad)).T.tolist())
    
    for j in delta_3d:
        
        band1 = S0 * np.exp(-0.5 * np.sqrt(t))  if j <= 0.5 else S0
        band2 = S0 * np.exp(0.5 * np.sqrt(t)) if j >= 0.5 else S0
        
        # VV method's strike
        res = so.fsolve(lambda x: delta(S0, x, r, d, sigma_vv(S0, x, r, d, sigma_atm, c_25, c_75, t), t, typ = 'C') - j, x0 = S0, maxfev=200 )
        # first approximation's strike
        res0 = so.fsolve(lambda x: delta(S0, x, r, d, first_approx(S0, r, d, t, x, c_75, sigma_atm, c_50), t, typ = 'C') - j, x0 = S0, maxfev=200 )
        #secondapproximation's strike
        res1 = so.fsolve(lambda x: delta(S0, x, r, d, second_approx(S0, x, c_25, sigma_atm, c_75, r, d, t), t, typ = 'C') - j, x0 = S0, maxfev=200 )
        
        # backprice market volatility
        sig = so.fsolve(lambda x: delta(S0, res, r, d, x, t, typ = 'C') - j, x0 = sigma_atm, maxfev=200)
        sig0 = so.fsolve(lambda x: delta(S0, res0, r, d, x, t, typ = 'C') - j, x0 = sigma_atm, maxfev=200)
        sig1 = so.fsolve(lambda x: delta(S0, res1, r, d, x, t, typ = 'C') - j, x0 = sigma_atm, maxfev=200)
        
        # VV method extracted market vol
        list_3d.append([t, j, sig[0]])
        # VV second approximation method
        list2nd_3d.append([t, j, sig1[0]])
        # VV 1st approximation
        list1st_3d.append([t, j, sig0[0]])


    
#t = 6/365
block = np.array([[(i,j,k) for j,k in  tem_3d_surface(i, list_3d)] for i in np.linspace(1/365, 1, 99)]).reshape(-1,3)
first_block = np.array([[(i,j,k) for j,k in  tem_3d_surface(i, list1st_3d)] for i in np.linspace(1/365, 1, 99)]).reshape(-1,3)
second_block = np.array([[(i,j,k) for j,k in  tem_3d_surface(i, list2nd_3d)] for i in np.linspace(1/365, 1, 99)]).reshape(-1,3)
quad_block = np.array([[(i,j,k) for j,k in  tem_3d_surface(i, list_quad)] for i in np.linspace(1/365, 1, 99)]).reshape(-1,3)


z_diff = block[:,2] - quad_block[:,2]
#z2_diff = block[:,2] - second_block[:,2]

#Delta lvl
x = np.linspace(0.01, 0.99, 99)
#Time in year
y = np.linspace(1/365, 1, 99)

# Plot in plotly library
#z = block[:,2].reshape(len(y), len(x))

#fig = go.Figure(data=[go.Surface(z=z_diff.reshape(len(y), len(x)), x=x, y=y)])
#fig = go.Figure(data=[go.Surface(z=block[:,2].reshape(len(y), len(x)), x=x, y=y)])
#fig.update_layout(title='VV versus Quad', scene = dict(
#                    xaxis_title='Delta',
#                    yaxis_title='Time in year',
#                    zaxis_title='Volatiity surface'))
#fig.show()

#fig = go.Figure(data=[go.Surface(z=z2_diff.reshape(len(y), len(x)), x=x, y=y)])
#fig = go.Figure(data=[go.Surface(z=block[:,2].reshape(len(y), len(x)), x=x, y=y)])
#fig.update_layout(title='VV versus seconds', scene = dict(
#                    xaxis_title='Delta',
#                    yaxis_title='Time in year',
#                    zaxis_title='Volatiity surface'))
#fig.show()


# Plot Quadratic Interpolation 3D volatility Surface
#zline = quad_block[:,2].reshape(len(y), len(x))

# Plot difference in volatility between VV method and quadratic interpolation
zline = z_diff.reshape(len(y), len(x))
xline = x
yline = y
fig = plt.figure()

Y, X = np.meshgrid(yline, xline)
Z = zline

ax = plt.axes(projection='3d')
ax.plot_surface(Y, X, Z,rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('ABS (VV - Quadratic)')
ax.set_xlabel('Delta')
ax.set_ylabel('Time in Years')
ax.set_zlabel('Volatility')
plt.show()

plt.plot(range(99), block[:99,2], label='VV')
plt.plot(range(99), first_block[:99,2], label='1st')
plt.plot(range(99), second_block[:99,2], label='2nd')
plt.plot(range(99), quad_block[:99,2], label='Quadratic')
plt.ylim(0.05, 0.14)
plt.legend()
plt.show()
