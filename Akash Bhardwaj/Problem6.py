import numpy as np
import matplotlib.pyplot as plt


L_p = -1
L = 1
T = 1
N = 100  
x = np.linspace(L_p, L, N)  
epsilon = 1e-3  
theta_values = [0, 0.5, 1]
mu_values = [0.1, 0.5, 0.9]


dt_values = np.linspace(0.001, 1.5, 5)  


def phi_0(x):
    return np.ones_like(x)

def phi_1(x):
    return x

def phi_2(x):
    return (3 / 2) * x**2 - (1 / 2)

def phi_3(x):
    return (5 / 2) * x**3 - (3 / 2)

phi_funcs = [phi_0, phi_1, phi_2, phi_3]


np.random.seed(0)  
b = np.random.uniform(0, 1, 4) 


def initial_condition_case1(x):
    return (1 - x)**4 * (1 + x)

def initial_condition_case2(x):
   
    polynomial_sum = np.sum([b[k] * phi_funcs[k](x) for k in range(4)], axis=0)
    C = max(0, -np.min(polynomial_sum))  
    return (1 - x)**4 * (1 + x) * (polynomial_sum + C)


def explicit_scheme(initial_condition, T, dt, theta, mu):
    num_time_steps = int(T / dt) + 1
    u = np.zeros((num_time_steps, N))
    u[0, :] = initial_condition(x)

    n = 0
    while n < num_time_steps - 1:
        for i in range(1, N - 1):
            u[n + 1, i] = (1 - theta) * u[n, i] + theta * (
                mu * u[n, i - 1] + (1 - 2 * mu) * u[n, i] + mu * u[n, i + 1]
            )
        
        u[n + 1, 0] = u[n + 1, -1] = 0  
        
        n += 1  

    return u




plt.figure(figsize=(15, 15))


for i, (theta, mu) in enumerate(zip(theta_values, mu_values)):
   
    plt.subplot(len(theta_values), len(mu_values), i + 1)

    for dt in dt_values:
        
        u_case1 = explicit_scheme(initial_condition_case1, T, dt, theta, mu)
        u_case2 = explicit_scheme(initial_condition_case2, T, dt, theta, mu)

       
        time_points = [0, 0.25,0.75, 1]
        time_indices = [int(tp / dt) for tp in time_points]

        
        for idx in time_indices:
           
            line, = plt.plot(x, u_case1[idx], label=f'dt={dt:.3f}', alpha=0.7)
            
            
    plt.title(f'Case 1: θ={theta}, μ={mu}')
    plt.xlabel('x')
    plt.ylabel('u(x, t)')
    plt.grid()

plt.tight_layout()
plt.show()


plt.figure(figsize=(15, 15))


for i, (theta, mu) in enumerate(zip(theta_values, mu_values)):
    
    plt.subplot(len(theta_values), len(mu_values), i + 1)

    for dt in dt_values:
       
        u_case2 = explicit_scheme(initial_condition_case2, T, dt, theta, mu)

        time_indices = [int(tp / dt) for tp in time_points]

    
        dt_count = 0 
        for idx in time_indices:
         
            line, = plt.plot(x, u_case2[idx], label=f'dt={dt:.3f}', alpha=0.7)
            if dt_count < 3:  
                plt.legend([line], [f'dt={dt:.3f}'], loc='best')
                dt_count += 1
            
    plt.title(f'Case 2: θ={theta}, μ={mu}')
    plt.xlabel('x')
    plt.ylabel('u(x, t)')
    plt.grid()

plt.tight_layout()
plt.show()


