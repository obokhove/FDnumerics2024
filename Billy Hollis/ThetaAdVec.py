# Billy Hollis 201421513


import numpy as np
import matplotlib.pyplot as plt

J = 400
L = 1
Lp = -1
dx = (L-Lp)/J
dt = 0.005
theta = 0.5
mu = dt / (dx)**2
epsilon = 10e-03
a = 1
gamma = dt / dx


T = 1

t = 0 



C1 = (1-theta) * epsilon * mu 
C2 = 1 + ((theta -1) * (a * gamma + 2 * epsilon * mu))
C3 = (1 - theta) * (a * gamma + epsilon * mu)
C4 = - (theta * epsilon * mu)
C5 = 1 + theta * (a * gamma + 2 * epsilon *  mu)
C6 = - theta * ( a * gamma + epsilon * mu)

A = np.zeros((J+1,J+1))

A[0,0] = 1 
A[-1,-1] = 1

for i in range(1,J):
    for j in range(i-1,i):
        A[i,j] = C4
    for j in range(i,i+1):
        A[i,j] = C5
    for j in range(i+1,i+2):
        A[i,j] = C6

I = np.zeros((J+1,1))
print(A)
I[0] = 0
I[-1] = 0

for i in range(1,J):
    I[i] = ((1 - (Lp +(i * dx))) ** 4 ) * (1 + (Lp + (i * dx)))
    
print(I)
    


b = np.zeros((J+1,1))
    
for j in range(1,J):
    b[j] = C1 * I[j-1] + C2 * I[j] + C3 * I[j+1]
    
for T in [0,0.25,0.5,0.75,1]:
    
    while t <= T:
        t = t
        x = np.linalg.solve(A,b)
        for j in range(1,J):
             b[j] = (C1 * x[j-1]) +(C2 * x[j]) + (C3 * x[j+1])
        xnew = x
        t = t + dt 
    X = np.linspace(Lp,L,J+1)
    Y = xnew 
    plt.plot(X,Y,label = f'After {T} Seconds')
    print(Y)

plt.xlim([Lp,L])
plt.ylim([0,3])
plt.grid()
plt.title(f'\u0394t = {dt}, \u03B8 = {theta}, \u03B5 = {epsilon}, J = {J} ')
plt.xlabel('Position x')
plt.ylabel('Approximate value of u')

plt.legend()
plt.show()
    

    

    





     



     
