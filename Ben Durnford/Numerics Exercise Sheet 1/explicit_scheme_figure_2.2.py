import numpy as np
import matplotlib.pyplot as plt

#define dims of box
L_p=0.0
L=1.0

#define spacetime mesh
N=50 #number of timesteps
J=20 #number of elements in x
dx=0.05
dt=0.0012
mu=dt/(dx**2)

t_n=N*dt

#define initial function u_0x
def u_0(x):
    y=np.zeros(len(x))
    for i,val in enumerate(x):
        if val <=0.5:
            y[i]+=2*val
        elif val>=0.5:
            y[i]+=2-(2*val)
    return y

#initiates the velocity at time 0
U=np.zeros(J+1)
x=np.array([i*dx for i in range(0,J+1)])
A=np.zeros(J+1)
U=u_0(x)
timesteps=np.array([0,1,25,50])
data=np.zeros((len(timesteps),J+1))
data[0,:]=U

#iteratively updates the velocity by using an explicit euler algorithm

for n in range(0,N):
    g=n+1 #the true effective timestep
    for j in range(1,J):
        A[j]=U[j]+mu*(U[j+1]-2*U[j]+U[j-1])  #A is used so that the values of U in the previous timestep are not updated within the calculation
    U[:]=A[:]
    if g in timesteps: #my code for extracting the data out of the for loop to plot it
        ind=np.where(timesteps==g)
        data[ind,:]=U

#plots the data
colors = plt.cm.viridis(np.linspace(0.1,0.9,len(timesteps)))  #the colour scheme of the graph
fig2,ax2=plt.subplots()
for i in range(0,len(timesteps)):
    ax2.plot(x,data[i,:],color=colors[i],marker='x',label=f'{timesteps[i]}')
    #ax2.scatter(x,data[i,:],color=colors[i],marker='x')

ax2.set(ylim=(-0.02,1.02),xlim=(-0.05,1.05),ylabel='Fluid velocity, U',xlabel='Position, x')
ax2.legend(title='Number of timesteps',frameon=False)
plt.show()
