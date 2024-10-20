import numpy as np
import matplotlib.pyplot as plt

L = 1.0             
Lp = 0.0           
epsilon = 0.1      
a_t = 1.0           
T = 1.0             
dt = 0.0012         
theta = 1

delta_x = [2,3,5,7,9,10] # different spatial resolution

def initial_condition(x):
    u0 = np.zeros_like(x)   
    u0[(x >= 0) & (x <= 0.5)] = 2 * x[(x >= 0) & (x <= 0.5)]  
    u0[(x > 0.5) & (x <= 1)] = 2 - 2 * x[(x > 0.5) & (x <= 1)]  
    return u0

def solve_advection_diffusion(delta_x, T, dt, theta, epsilon, L, Lp, a_t):  
    x = np.linspace(Lp, L, delta_x)  
    dx = x[1] - x[0] 
    u = initial_condition(x)  
    u_new = np.zeros(delta_x)  
    num_steps = int(T / dt)
    for n in range(num_steps):  
        A = np.zeros((delta_x, delta_x))  
        b = np.zeros(delta_x)       
       
        for i in range(1, delta_x-1):
            A[i, i-1] = -theta * dt * a_t / (2 * dx)  
            A[i, i] = 1 + (epsilon * dt / dx**2)      
            A[i, i+1] = -theta * dt * a_t / (2 * dx)  
        A[0, 0] = A[-1, -1] = 1  
        b[0] = b[-1] = 0         
        for i in range(1, delta_x-1):
            b[i] = u[i] + (1 - theta) * dt * (
                a_t * (u[i+1] - u[i-1]) / (2 * dx) + 
                epsilon * (u[i+1] - 2 * u[i] + u[i-1]) / (dx**2)
            )
        
        u_new = np.linalg.solve(A, b)
        u = u_new.copy()  
    return x, u  

plt.figure(figsize=(10, 6))

for N in delta_x:
    x, u = solve_advection_diffusion(N, T, dt, theta, epsilon, L, Lp, a_t)
    plt.plot(x, u, label=f'delta_x = {N}') 


plt.title('Advection-Diffusion Solution at Time T with Fixed dt = 0.0012')
plt.xlabel('x')  
plt.ylabel('u(x, T)') 
plt.legend()  
plt.grid()  
plt.show()  
