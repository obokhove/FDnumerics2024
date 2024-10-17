# Billy Hollis 201421513

import numpy as np
import matplotlib.pyplot as plt

J = 20
dx = 1 / J
dt = 0.0013
T = 0
mu = dt / (dx) ** 2 
t = 0 

U = np.zeros((J+1))



for j in range(0,J//2+1):
    U[j] = 2 * j * dx
for j in range(J//2+1,J+1):
    U[j] = 2 - (2 * j * dx)
    

for T in [0,0.01,0.02,0.03,0.04,0.05]: 

    while t < T:
        t = t 
        Unew = np.zeros(J+1)
        Unew[0] = 0
        Unew[-1] = 0
        for j in range(1,J):
            Unew[j] = U[j] + mu *(U[j-1] - 2 * U[j] + U[j+1])
        U = Unew
        t = t + dt 
    X = np.linspace(0,1,J+1)
    Y = U 
    plt.plot(X,Y,label = f' After {T} Seconds')

plt.grid()
plt.xlim([0,1])
plt.ylim([0,1])
plt.title(f'\u0394t = {dt}, J = {J}')
plt.xlabel('Position x')
plt.ylabel('Approximate value of u')
plt.legend()
plt.show()






         



