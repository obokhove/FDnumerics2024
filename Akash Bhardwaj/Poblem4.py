
# Problem 4: Explicit scheme to reproduce fig 2.2 in M&M

import numpy as np
import matplotlib.pyplot as plt

J=20 # total number of grid points
L=1.0 #length of domain
dx= L/J # grid size
x=np.linspace(0,L,J +1)

dt1= 0.0012 # first time step size
dt2=0.0013 # second time step size
Nt= 50 # total number of time steps

 #indexing the grid point starting form 0 to J+1 (B.Cs) 

def I_C(x):
    u0= np.zeros_like(x)
    u0[x<=0.5]=2*x[x<=0.5]
    u0[x>0.5]=2-2*x[x>0.5]
    return u0

def Explicit_scheme(Nt,dt,dx,J):
    u=I_C(x)
    u_solutions=[]
    u_solutions.append(u.copy())
    
    for n in range(Nt):
        u_new =u.copy()
        
        for i in range(1,J):
            u_new[i] = u[i]+(dt/dx**2)*(u[i+1]-2*u[i]+u[i-1])
        u_new[0] = 0 # Boundary condition at LHS
        u_new[J] = 0 # Boundary condition at RHS
        u = u_new.copy()
        if n in [0, 10, 20, 30, 40, 50]:
            u_solutions.append(u.copy())
                
    return u_solutions

def plot_results():
    plt.figure(figsize=(10, 4))
   
    u_solutions_dt1 = Explicit_scheme(Nt, dt1, dx, J)
    
    plt.subplot(1, 2, 1)
    plt.plot(x, u_solutions_dt1[0], 'k-', label='t=0 (Initial condition)', marker='o')  
    plt.plot(x, u_solutions_dt1[1], 'r-', label='After 10 time step', marker='o') 
    plt.plot(x, u_solutions_dt1[2], 'g-', label='After 20 time steps', marker='o')  
    plt.plot(x, u_solutions_dt1[3], 'b-', label='', marker='o')  
    plt.plot(x, u_solutions_dt1[4], 'o-', label='', marker='o')  
    plt.plot(x, u_solutions_dt1[5], 'y-', label='After 50 time steps', marker='o')  
    plt.title(r'$\Delta t = 0.0012$ (Stable)', fontsize=12)  
    plt.xlabel('x') 
    plt.ylabel('u')  
    plt.legend()  
    plt.grid(True)
    
    u_solutions_dt2 = Explicit_scheme(Nt, dt2, dx, J)
    
    plt.subplot(1, 2, 2)
    plt.plot(x, u_solutions_dt2[0], 'k-', label='t=0 (Initial condition)', marker='o')  
    plt.plot(x, u_solutions_dt2[1], 'r-', label='After 10 time step', marker='o')
    plt.plot(x, u_solutions_dt2[2], 'g-', label='After 20 time steps', marker='o') 
    plt.plot(x, u_solutions_dt2[3], 'b-', label='', marker='o') 
    plt.plot(x, u_solutions_dt2[4], 'o-', label='', marker='o') 
    plt.plot(x, u_solutions_dt2[5], 'y-', label='After 50 time steps', marker='o')  
    plt.title(r'$\Delta t = 0.0013$ (Instable)', fontsize=12)  
    plt.xlabel('x')  
    plt.ylabel('u')  
    plt.legend()  
    plt.grid(True)  

    plt.tight_layout()  
    plt.show()  
plot_results()