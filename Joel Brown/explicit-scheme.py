# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 19:33:27 2024

@author: mqhq8337
"""

import matplotlib.pyplot as plt
from numpy import random
import numpy as np

T = 1  #Given by the domain of t [0,T]
Lp = -1 #Given by the domain of x [Lp,L]
L = 1
eps = 10E-3  #Diffusion constant given in equation (1)
J = 50  #Number of sections the x domain will be split into
N = 100  #Number of sections the t domain will be split into
dx = (L-Lp)/J  #Position step
dt = T/N  #Time step
mu = dt/(dx*dx)
random.seed(314) #Different seeds give different random values of b_k in the initial condition

#Legendre polynomials
def phi_0(x):
   if Lp < x < L:
       return 1
def phi_1(x):
   if Lp < x < L:
       return x
def phi_2(x):
   if Lp < x < L:
       return (3/2)*x*x-(1/2)
def phi_3(x):
   if Lp < x < L:
       return (5/2)*x*x*x-(3/2)*x

#Assigning random values to b_k in range [0,1] according to normal distribution
b = 4*[0]
for k in range(0,4):
    b[k] = random.random_sample()
print(b)

#Defining the initial conditions
def u0_1(x) :
   if Lp < x < L:
       return ((1-x)**4)*(1+x)
def phi(x):
    if Lp < x < L:
        return b[0]*phi_0(x) + b[1]*phi_1(x) + b[2]*phi_2(x) + b[3]*phi_3(x)
    
def u0_2(x) :
   if Lp < x < L:
       mini = np.min(phi(x))
       #C is defined to ensure that phi(x) + C is positive for x in [Lp,L]
       if mini <= 0:
           C = -mini
       else:
           C = 0
       return ((1-x)**4)*(1+x)*(phi(x) + C)

#Defining the advection coefficient as a(t)=1
def a(t) :
    if t >= 0:
        return 1


u = []  #Value of U_j^0, this will be updated at each time step
f = []  #Recursive relation which updates at each node, U_j^(n+1) = f_j
x = []  #Array of x-values to plot the results
for j in range(0,J+1):
    x.append(Lp + j*dx)  
    u.append(u0_2(x[j]))  #u is given the initial condition at t=0, change between u0_1 and u0_2 for different conditions
    f.append(0)
ustart = u[:] #Plot of u after zero time steps

#Beginning with array u at time t_0, we replace the values at each time step and take out results we wish to see, until time t_N=T
t = 0
while t < T+dt:
    u[0] = u[J] = 0  #Setting boundary conditions
    f[0] = 0
    #Evaluate all f_j in a row
    for j in range(1,J):
        f[j] = u[j]+a(t)*dx*mu*(u[j+1]-u[j])+eps*mu*(u[j+1]-2*u[j]+u[j-1])
    #Evaluate all u_j in a row
    for j in range(1,J):
        u[j] = f[j] 
    #Here we save the values of the array u at times t=0.01,0.25,0.5,1 to display on plots, allowing for small computational errors
    if 0.999*dt < t < 1.001*dt: 
            u1 = u[:]
    elif 24.999*dt < t < 25.001*dt:
        u25 = u[:]
    elif 49.999*dt < t < 50.001*dt: 
            u50 = u[:]
    elif 99.999*dt< t < 100.001*dt:
        u100 = u[:]
    t = t + dt #Process is repeated for the next time step until t=T


#The results are displayed on a single plot for times t=0,0.01,0.25,0.5,1
fig,ax=plt.subplots()
ax.plot(x,ustart,marker='o',label='t=0.00')
ax.plot(x,u1,marker='o', label='t=0.01')
ax.plot(x,u25,marker='o', label='t=0.25')
ax.plot(x,u50,marker='o', label='t=0.50')
ax.plot(x,u100,marker='o', label='t=1.00')
plt.xlabel('x')
plt.ylabel('u')
ax.legend(frameon=False)
plt.show()


