import numpy as np
import matplotlib.pyplot as plt
import random

np.random.seed(11)

#defines global parameters
L=-1
L_p=1  #the dimensions of the box
N=100 #number of timesteps
J=20000 #number of elements x is split into
dx=(L_p-L)/J  #size of each x element
print(f'dx is {dx}')
#dt=0.0013     #size of each timestep
T=1.0
dt=T/N
mu=dt/(dx**2)
print(f'mu is {mu}')
theta=1.0
beta=theta*mu   
epsilon=1.0E-0  #the size of the diffusion coefficient

#define the initial conditions and a(t) as functions

def spike(x):  #function which is linear and symmetric around its midpoint
    y=np.zeros(len(x))
    for i,val in enumerate(x):
        if val <=0.5:
            y[i]+=2*val
        elif val>=0.5:
            y[i]+=2-(2*val)
    return y

def f_1(x):
    y=(1-x)**4 * (1+x)
    return y

def f_2(x):
    y=(1-x)**4 * (1+x)
    b=np.random.random_sample(4)
    print(f'The b values are {b}')
    leg_0=1*b[0]
    leg_1=x*b[1]
    leg_2=(1.5*x**2-0.5)*b[2]
    leg_3=(2.5*x**3-1.5*x)*b[3]
    leg_sum=leg_0+leg_1+leg_2+leg_3
    C=np.min(leg_sum)
    if C<=0:
        leg_sum-=C
    fn=y*leg_sum
    return fn

def a(t):  
    return 0.0

#initiate the velocity field at time 0

U=np.zeros(J+1) 
x=np.array([L+i*dx for i in range(0,J+1)])  #x coordinates
A=np.zeros(J+1)
e=np.zeros(J+1)
f=np.zeros(J+1)
U=f_2(x)
print(U)


times = np.linspace(0,T,5) #saves the data for graphing
data=np.zeros((len(times),J+1))
cond=[True for i in range(len(times))]
data[0,:]=U
print(U)

#finite difference algorithm which is based on the Thomas algorithm
t=0
i=0
m=1
while t<=T:
    t+=dt
    e[0]=0.0
    f[0]=0.0
    for j in range(1,J):
        dnm=(1+beta*(2*epsilon-epsilon*e[j-1]+a(t+dt)*dx))
        e[j]=(theta*mu*(epsilon+a(t+dt)*dx))/dnm
        fnum=U[j]+mu*(1-theta)*(epsilon*(U[j+1]-2*U[j]+U[j-1])+dx*a(t)*(U[j+1]-U[j]))+mu*epsilon*theta*f[j-1]
        f[j]=fnum/dnm


    for j in reversed(range(1,J)):
        U[j]=f[j]+e[j]*U[j+1]
    
    #ind=np.where((t>times))
    #print(ind)
    #ind=ind[0]
    #print(ind)
    if t>times[m]:
        data[m,:]+=U[:]
        m+=1
    
#graphs the data

colors = plt.cm.viridis(np.linspace(0.1,0.9,len(times)))
fig2,ax2=plt.subplots()
for i in range(0,len(times)):
    #ax2.plot(x,data[i,:],color=colors[i],marker='x',label=f'{round(times[i],2)}')
    ax2.plot(x,data[i,:],color=colors[i],label=f'{round(times[i],2)}')
    
ax2.set(ylabel='Fluid velocity, U',xlabel='Position, x',xlim=(-1.0,1.0),ylim=(0.0,2.5))
ax2.legend(title='Time , t',frameon=False, loc='upper left',ncols=2)
plt.show()


#print(f_2(x))
#print(np.min(f_2(x)))

#Stability condition graph

