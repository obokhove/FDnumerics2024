# Billy Hollis 201421513

import numpy as np
import matplotlib.pyplot as plt

J = 40
L = 1
Lp = -1
dx = (L-Lp) / J 
dt = 0.0013
T = 0
mu = dt / (dx) ** 2 
t = 0 
gamma =  dt / (dx) 
a = 1
epsilon = 10e-03


C1 = epsilon * mu
C2 = 1 - 2 * (epsilon * mu) - (a * gamma)
C3 = (a * gamma) + (epsilon * mu)

U = np.zeros(J+1)

for j in range(0,J+1):
    U[j] = ((1 - (Lp +(j * dx))) ** 4 ) * (1 + (Lp + (j * dx)))
    
for T in [0,0.25,0.5,0.75,1]:
    
    
    while t <= T:
        t=t
        Unew = np.zeros(J+1)
        Unew[0] = 0
        Unew[-1] = 0
        for j in range(1,J):
            Unew[j] = C1 * U[j-1] + C2 * U[j] + C3 * U[j+1]
        U = Unew
        t = t + dt
    X = np.linspace(Lp,L,J+1)
    Y = U 
    print(U)
    plt.plot(X,Y,label = f' After {T} Seconds')
    
plt.grid()
plt.xlim([Lp,L])
plt.ylim([0,3])
plt.title(f'\u0394t = {dt}, J={J}')
plt.xlabel('Position x')
plt.ylabel('Approximate value of u')
plt.legend()
plt.show()



        
        
    
   


