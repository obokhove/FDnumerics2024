# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 23:00:45 2024

@author: pm20ga
"""

import numpy as np
import matplotlib.pyplot as plt

def q4(dt):
    J = 20
    x = np.linspace(0,1, J+1)
    dx=0.05
    T= 50
    
    #define U0
    U0 = np.zeros(J+1)
    for J, i  in enumerate(x):
        if  0<=i<=1/2:
            U0[J] = 2*i
        elif 1/2<=i<=1:
            U0[J] = 2-(2*i)
    print(U0)

            
    #then need to find a way to update a function U
    #first define a function U
    Un = U0 #define an old vector of Un
    Un_plus1 = np.zeros(J+1)
    #BCs
    Un_plus1[0]=0
    Un_plus1[J]=0
    
    plt.figure()
    plt.plot(x,U0, 'mo-', label = 't=0')
    
    for n in range(1,T+1):
        for j in range(1, J):
            Un_plus1[j] = Un[j] + (dt / (dx ** 2)) * (Un[j + 1] - (2 * Un[j]) + Un[j - 1])
        if n ==1:
            plt.plot(x,Un_plus1, 'bo-', label='first timestep')
            print(Un_plus1)
        elif n==25:
            plt.plot(x,Un_plus1,'ro-', label = '25th timestep')
        elif n==50:
            plt.plot(x,Un_plus1,'yo-', label='50th timestep')
        Un = Un_plus1.copy()
    
    plt.title(f"dt = {dt}")
    plt.xlabel('x')
    plt.ylabel('U')
    plt.legend()
    plt.savefig(f"gulana_q4_dt_{dt}.jpeg")
    

q4(0.0012)
q4(0.0013)

    
    