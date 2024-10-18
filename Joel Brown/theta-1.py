# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 13:12:15 2024

@author: mqhq8337
"""

import matplotlib.pyplot as plt

T = 1  #Given by the domain of t [0,T]
Lp = -1 #Given by the domain of x [Lp,L]
L = 1
th = 0  #Value of theta in the theta scheme
#th = 0 gives the explicit scheme, th = 1 gives the implicit schmeme
eps = 10E-3  #Diffusion constant given in equation (1)
J = 50  #Number of sections the x domain will be split into
N = 100  #Number of sections the t domain will be split into
dx = (L-Lp)/J  #Position step
dt = T/N  #Time step
mu = dt/(dx*dx)
print(mu)

#Defining the initial condition
def u0(x) :
   if Lp < x < L:
       return ((1-x)**4)*(1+x)

#Defining the advection coefficient as a(t)=1
def a(t) :
    if t >= 0:
        return 1


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
t = 0
while t < T+dt:
    u[0] = u[J] = 0  #Setting boundary conditions
    e[0] = 0
    f[0] = 0
    #Evaluate all e_j and f_j in a row
    for j in range(1,J):
        den = 1 + th*mu*(a(t+dt)*dx+2*eps) - th*mu*eps*e[j-1]
        enum = th*mu*(a(t+dt)*dx+eps)
        fnum = u[j] + (1-th)*mu*(a(t)*dx*(u[j+1]-u[j])+eps*(u[j+1]-2*u[j]+u[j-1])) + th*mu*eps*f[j-1]
        e[j] = enum/den
        f[j] = fnum/den
    #Evaluate all u_j in a row "backwards" since U_j depends on U_(j+1)
    for j in range(J-1,0,-1):
        u[j] = e[j]*u[j+1] + f[j] 
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


