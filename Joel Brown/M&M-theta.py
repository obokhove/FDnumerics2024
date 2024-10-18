# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 14:00:31 2024

@author: mqhq8337
"""

import matplotlib.pyplot as plt

Lp = 0 #Given by the domain of x [Lp,L]
L = 1
th = 0  #Value of theta in the theta scheme
#th = 0 gives the explicit scheme, th = 1 gives the implicit schmeme
J = 20  #Number of sections the x domain will be split into
N = 100  #Number of sections the t domain will be split into
dx = (L-Lp)/J  #Position step
dt = 0.0012  #Time step
mu = dt/(dx*dx)
print(mu)

#Defining the initial condition
def u0(x) :  
    if Lp <= x < Lp + 0.5*(L-Lp) :
        return 2*x
    elif Lp + 0.5*(L-Lp) <= x <= 1 :
        return 2 - 2*x

u = []  #Value of U_j^0, this will be updated at each time step
e = []  #Recursive relation which updates at each node, U_j^(n+1) = e_j*U_(j+1)^(n+1) + f_j
f = []
x = []  #Array of x-values to plot the results
for j in range(0,J+1):
    x.append(Lp + j*dx)  
    u.append(u0(x[j]))  #u is given the initial condition at t=0
    e.append(0)
    f.append(0)
ustart = u[:] #Plot of u after zero time steps

#Beginning with array u at time t_0, we replace the values at each time step and take out results we wish to see, until time t_N=T
for n in range(1,N+1):
    u[0] = u[J] = 0  #Setting boundary conditions
    e[0] = 0
    f[0] = 0
    #Evaluate all e_j and f_j in a row
    for j in range(1,J):
        den = 1+2*th*mu-th*mu*e[j-1]
        enum = th*mu
        fnum = u[j]+(1-th)*mu*(u[j+1]-2*u[j]+u[j-1])+th*mu*f[j-1]
        e[j] = enum/den
        f[j] = fnum/den
    #Evaluate all u_j in a row "backwards" since U_j depends on U_(j+1)
    for j in range(J-1,0,-1):
        u[j] = e[j]*u[j+1] + f[j] 
     #Here we save the values of the array u at time steps n=1,10,20,25,30,40,50 to display on plots
    if n == 10:
         u10 = u[:]
    elif n == 20:
         u20 = u[:]
    elif n == 30:
         u30 = u[:]
    elif n == 40:
         u40 = u[:]
    elif n == 50: 
            u50 = u[:]



fig,ax=plt.subplots()


#Plot multiple time-profiles in one plot
ax.plot(x,ustart,marker='o',label='After 0 steps')
ax.plot(x,u10,marker='o', label='After 10 step')
ax.plot(x,u20,marker='o', label='After 20 steps')
ax.plot(x,u30,marker='o', label='After 30 steps')
ax.plot(x,u40,marker='o', label='After 40 steps')
ax.plot(x,u50,marker='o', label='After 50 steps')
plt.xlabel('x')
plt.ylabel('u')
ax.legend(frameon=False)
plt.show()

