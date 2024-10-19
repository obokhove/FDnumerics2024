# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 21:15:27 2024

@author: pm20ga
"""

import numpy as np
import matplotlib.pyplot as plt

def phi(k, x):
    if k == 0 :
        return np.ones_like(x)
    elif k == 1:
        return x
    elif k ==2:
        return 0.5* (3* (x**2) -1)
    elif k == 3:
        return 0.5 * (5*x **3 - 3 *x)

def boundary(x,J):
    #function for determining boundary conditions
    bk = [0.7703829,  0.57331959, 0.86436473, 0.82221663]
    print(bk)
    UBC = np.zeros(J+1)
    first_term = (1-x)**4 * (1+x)
    second_term = sum(bk[k]* phi(k, x) for k in range(4))
    UBC = first_term*second_term
    C = max(0, -np.min(second_term))
    
    return UBC + C


def q5(Lp,L, a, J, T, theta, epsilon, CFL, N):
    dx = (L-Lp)/J
    print(f'dx={dx}')
    dt = T/N
    print(f'dt={dt}')
    At = (a*dt)/dx
    B = (epsilon*dt)/((dx)**2)
    mu = dt/(dx**2)
    print(f'mu={mu}')
    #discretise x
    x = np.linspace(Lp,L, J+1)
    
    #create a matrix to store the U values (A in Ax=b)
    A = np.zeros((J+1, J+1))
    
    #create a vector to store all the u values in Ax=b
    x_matrix = np.zeros(J+1).transpose()
    
    #create an empty vector B
    b = np.zeros(J+1).transpose()
    
    #define the boundary conditions
    UBC = boundary(x,J)
    U0 = UBC.transpose()
    #print(U0)
    for i in range(0, J+1):
        if i==0 or i == J:
            U0[i]=0
        else:
            U0[i]=UBC[i]
    plt.figure()
    plt.plot(x,U0, 'm', label = 'time(s) = 0')
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
     
    eigenvalues = np.linalg.eigvals(A)
    print("Eigenvalues of matrix A:", eigenvalues)
    
    Uold = U0.copy()  
    time = 0
    a_timestep = np.linspace(0, T, 5)
    print(a_timestep)
    counter = 0 
    
    while time <= T:
        time+=dt
        for i in range(1,J):
            b[i] = Uold[i+1]*(At*(1-theta)+B*(1-theta))+Uold[i]*(1-At*(1-theta)-2*B*(1-theta))+Uold[i-1]*(B*(1-theta))
            x_matrix = np.linalg.solve(A, b)
        Uold = x_matrix.copy() 
        if time>a_timestep[counter]:
            plt.plot(x, x_matrix, label = f'time(s) = {time:.3f}')
            plt.title(f'dx={dx}')
            plt.xlabel('x')
            plt.ylabel('U')
            plt.xlim(-1.0, -0.6)
            plt.legend()
            plt.savefig(f"gulana_q7_dx_{dx}_final.jpeg")
            counter+=1
        
        
q5(-1,1,1,5,1,1,0.001,1, 1000000)
