# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 16:59:23 2024

@author: pm20ga
"""

import numpy as np
import matplotlib.pyplot as plt

def boundary(x,J):
    #function for determining boundary conditions
    UBC = np.zeros(J+1)
    for J, i  in enumerate(x):
        if  0<=i<=1/2:
            UBC[J] = 2*i
        elif 1/2<=i<=1:
            UBC[J] = 2-(2*i)
    return UBC

def q5(Lp,L, dt, a, dx, T, theta, epsilon):
    #where T = number of timesteps
    At = (a*dt)/dx
    B = (epsilon*dt)/((dx)**2)
    #discretise x
    J = int((L-Lp)/dx)
    x = np.linspace(Lp,L, J+1)
    CFL = (dt*a)/dx
    
    #create a matrix to store the U values (A in Ax=b)
    A = np.zeros((J+1, J+1))
    
    #create a vector to store all the u values in Ax=b
    x_matrix = np.zeros(J+1).transpose()
    
    #create an empty vector B
    b = np.zeros(J+1).transpose()
    
    #define the boundary conditions
    UBC = boundary(x,J)
    U0 = UBC.transpose()
    for i in range(0, J+1):
        if i==0 or i == J:
            U0[i]=0
        else:
            U0[i]=UBC[i]

    #assign the value of the middle diagonal
    middle_val = (1+(At*theta)+(B*theta))
    middle_diagonal = np.full(J+1, middle_val)
    
    #assign the value of the upper diagonal
    upper_val= -((At*theta)+(B*theta))
    upper_diagonal = np.full(J, upper_val)
    
    #assign the value of the lower diagonal
    lower_val = -(B*theta)
    lower_diagonal = np.full(J, lower_val)
    A = np.diag(middle_diagonal) + np.diag(upper_diagonal, k=1)+ np.diag(lower_diagonal, k=-1)
    
    #now I want to add a row of 1,000000 to the top row
    A[0,0]=1
    A[0,1]=0
    A[-1,-1]=1
    A[-1,-2]=0
     
    plt.figure()
    plt.plot(x,U0, 'ro-')
    
    Uold = U0.copy()  
    time = 0
    for n in range(1, T+1):
        for i in range(1,J):
            b[i] = Uold[i+1]*(At*(1-theta)+B*(1-theta))+Uold[i]*(1-At*(1-theta)-2*B*(1-theta))+Uold[i-1]*(B*(1-theta))
            x_matrix = np.linalg.solve(A, b)
        Uold = x_matrix.copy() 
        if n == 1:
            plt.plot(x,x_matrix,'go-', label = 'timestep = 1')
        elif n == 3:
            plt.plot(x,x_matrix,'bo-', label = 'timestep = 3')
        elif n == 6:
            plt.plot(x,x_matrix, 'yo-', label = 'timestep = 6')
        elif n == 8:
            plt.plot(x,x_matrix, 'mo-', label=  'timestep = 8')
            plt.title(f'Advection when epsilon = 0 and CFL = {CFL}')
            plt.xlabel('x')
            plt.ylabel('U')
            plt.legend()
            plt.savefig(f"gulana_q5.jpeg")
            
            

q5(0,1,0.05,1,0.05,50,0,0) #make sure dt=dx for the CFL to = 1 when a(t) =1

