# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 16:59:55 2024

@author: pm20ga
"""

import numpy as np
import matplotlib.pyplot as plt

def q4(dt):
    J = 20
    x = np.linspace(-1,1, J+1)
    dx=0.1
    T= 100
    
    #define U0
    U0 = np.zeros(J+1)
    U0 =(1-x)**4 * (1+x)
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
            Un_plus1[j] = Un[j] + (dt / dx) * (Un[j + 1] - Un[j]) + (0.001 * dt / dx ** 2) * (Un[j + 1] - 2 * Un[j] + Un[j - 1])
        if n ==1:
            plt.plot(x,Un_plus1, 'bo-', label='t(s) = 0.01')
            print(Un_plus1)
        elif n == 100:
            plt.plot(x,Un_plus1, 'co-', label='t(s) = 1.00')
        elif n==25:
            plt.plot(x,Un_plus1,'ro-', label = 't(s) = 0.25')
        elif n == 50:
            plt.plot(x,Un_plus1, 'go-', label='t(s) = 0.50')
        elif n==75:
            plt.plot(x,Un_plus1,'yo-', label='t(s) = 0.75')
        Un = Un_plus1.copy()
    
    plt.title(f"Verification of the theta method of question 4 with explicit scheme")
    plt.xlabel('x')
    plt.ylabel('U')
    plt.legend()
    plt.savefig(f"gulana_q6_verification.jpeg")
    

q4(0.01)