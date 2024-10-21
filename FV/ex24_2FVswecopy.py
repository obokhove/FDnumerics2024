####################################################################################
#
# Numerical exercise on Finite Volume (FV) shallow-water dynamics
# 02/03-11-2023 by Onno Bokhove
# Sofar: linear and nonlinear case for periodic and specified boundary conditions
# Solves the problem at hand
# Run for CFL=0.9 (.e.g) and 1.0 as well as nbc=0,1 and Neqn=0,1; and, explore
####################################################################################
#
# GENERIC MODULES REQUIRED:
#
import numpy as np
import os
import errno
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import time
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec
from math import pi, e


def Hxxtopo(H0, W, L1, Hb, slope, xk12, Nb):
 if Nb<3: 
  return H0+0.0*xk12
 else:
  return H0*np.heaviside(W-xk12,0.5)+Hb*np.heaviside(xk12-L1,0.5)+(H0-slope*(xk12-W))*np.heaviside(xk12-W,0.5)*np.heaviside(L1-xk12,0.5);

plt.close("all")
#
# Define parameters
#
Nkel = 4000  # number of element
# Make cells and make cell edges/nodes
Xce = np.zeros(Nkel+1)
Kk  = np.zeros(Nkel)
xmk = np.zeros(Nkel) #  Nkel cells
u1  = np.zeros(Nkel) # Nkel u1's
u2  = np.zeros(Nkel) # Nkel u2's

#
# Run for CFL=0.9 (e.g.) and 1.0 as well as nbc=0,1 and Neqn=0,1 and watch what happens
#
Nbc = 3 # Nbc = 0: periodic; Nbc=1: solid walls; Nbc=2: prescribed via exact solution; Nbc=3: left incoming, outgoing, extrapolate
Nimpl = 1 #  when Neqn=0, Nimpl=1 is geometric flux and SE.
thetaa = 0.333
Neqn = 0     #  Neqn=0 mean linear SWE; Neqn=1 mwans nonlinear SWE
nlflux = 0
CFL = 1.0

gravr = 9.8
gtilde = gravr/gravr
if Nbc <3:
 Ld = 2.0*np.pi # length domain
 mint = 4
 H0s = 1.0;
 H0 = 1.0
 kw = pi*mint/Ld
 omeg = gtilde*kw
 ck = 0.2
 Nper = 10.0
elif Nbc==3:
 H0s = 40 # m
 H0 = 1
 Ld = 12
 L1 = 10
 W = 2
 Hb = 0.1
 U0s = np.sqrt(gravr*H0s)
 slope = (H0-Hb)/(L1-W)
 bslope = 1.0/50.0
 Ls = slope*H0s/bslope
 Ts = Ls/U0s #
 Tps = 6  #
 Tp = Tps/Ts
 omeg = 2*pi/Tp
 print('Ls, Ts',Ls,Ts)
 kw = omeg/np.sqrt(gtilde*H0)
 As = 1 # m
 Ad = As/H0s
 Cc = gtilde*Ad*kw/omeg
 Nper = 45

c0 = np.sqrt(gtilde*H0)
dxx = Ld/Nkel
#
F1  = np.zeros(Nkel+1)  # Flux for u1
F2  = np.zeros(Nkel+1)  # Flux for u2
Hx = np.zeros(Nkel+1)  # Flux for H(x) at cell edges (continuous function) at xce
F1r  = np.zeros(Nkel+1)  # 
F2r  = np.zeros(Nkel+1)  # 
F1l  = np.zeros(Nkel+1)  # 
F2l  = np.zeros(Nkel+1)  # 
Sl  = np.zeros(Nkel+1)  # estimate of nonlinear wave speed on left
Sr  = np.zeros(Nkel+1)  # estimate of nonlinear wave speed on right
SrmSl  = np.zeros(Nkel+1)  #  difference
U1l = np.zeros(Nkel+1)  # value u1 left of cell edge 
U1r = np.zeros(Nkel+1)  # value u1 right of cell edge 
U2l = np.zeros(Nkel+1)  # value u2 left of cell edge 
U2r = np.zeros(Nkel+1)  # value u2 right of cell edge 
U1star = np.zeros(Nkel+1) # Riemann state at cell edge or star state for u1 
U2star = np.zeros(Nkel+1) # Riemann state at cell edge or star state for u2
for jj in range(0,Nkel+1):   
 Xce[jj] = jj*dxx  # edge/node positions; add a jiggle to make nonuniform
# end for-loop
#  print('Xce, Kk, xmk:',Xce,Kk,xmk)  # Check
Kk[0:Nkel] = Xce[1:Nkel+1]-Xce[0:Nkel] # Nkel cell lengths
#  xmk[0:Nkel] = 0.5*(Xce[1:Nkel+1]+Xce[0:Nkel]) # Nkel cell middles
xmk[0:Nkel-1] = 0.5*(Xce[1:Nkel]+Xce[0:Nkel-1]) 
xmk[Nkel-1] = 0.5*(Xce[Nkel]+Xce[Nkel-1]) 
#
# print('Xce, Kk, xmk:',Xce,Kk,xmk)

# Plot initial condition
tijd = 0.0

Hx = H0+0.0*Xce
if Neqn==0: # Linear swe
 # periodic
 if Nbc==0:
  u1 = ck*np.cos(kw*xmk-omeg*tijd) # u1 is eta
  u2 = (omeg/kw)*ck*np.cos(kw*xmk-omeg*tijd) # u2 is velocity
  Hx = Hxxtopo(H0, 0.0, 0.0, 0.0, 0.0, Xce, Nbc)
 elif Nbc==1: # standing wave
  u1 = ck*np.cos(mint*pi*xmk/Ld)*np.cos(omeg*tijd) # u1 is eta
  u2 = (omeg*ck/H0)*(Ld/(mint*pi))*np.sin(mint*pi*xmk/Ld)*np.sin(omeg*tijd) # u2 is velocity
  Hx = Hxxtopo(H0, 0.0, 0.0, 0.0, 0.0, Xce, Nbc)
 elif Nbc==3: # rest flow
  u1 = 0*xmk
  u2 = 0.0*xmk
  Hx = Hxxtopo(H0, W, L1, Hb, slope, Xce, Nbc)
 # 
 c00 = np.sqrt(gtilde*Hx)
 
else: #  Nonlinear swe
 u1 = H0+ck*np.cos(kw*xmk) # h=H0+eta
 u2 = H0*(omeg/kw)*ck*np.cos(kw*xmk)
#  end Neqn
Tend = Nper*(2.0*np.pi/omeg)
#  u1[Nkel-1]=2.0 #   next few print lines are checks used at early stage programming
#  print('xmk:',xmk)
#  print('u1:',u1)
#  print('F1:',F1)
#  print('u1[0],u1[Nkel-1], Xce[Nkel], Xce[0]:',u1[0],u1[Nkel-1],Xce[Nkel],Xce[0])
plt.figure(1)
plt.subplot(211)
if Neqn==0: # linear swe
 plt.plot(xmk,H0+u1,'-',lw=2) # eta
 plt.plot(Xce,H0-Hx,'k',lw=3) # eta
 if Nbc==3:
  plt.plot(Xce,H0+Ad*(H0/Hx)**0.25,'--r',lw=2) # eta
else: # nonlinear swe
 plt.plot(xmk,u1,'-',lw=2)    # h=H0+eta
#  
plt.xlabel('$x$',fontsize=18)
plt.ylabel('$u_1=h(x,t)$',fontsize=18)
plt.subplot(212)
plt.plot(xmk,u2,'-',lw=2)
plt.xlabel('$x$',fontsize=18)
plt.ylabel('$u_2=u(x,t)$',fontsize=18)

if Neqn==0 and Nbc==3: # linear swe and beach
 plt.figure(2)
 plt.subplot(211)
 plt.plot(Ls*xmk,H0s*(H0+u1),'-',lw=2) # eta
 plt.plot(Ls*Xce,H0s*(H0-Hx),'k',lw=3) # eta
 plt.plot(Ls*Xce,H0s*(H0+Ad*(H0/Hx)**0.25),'--r',lw=2) # eta 
 plt.xlabel('$x$ (m)',fontsize=18)
 plt.ylabel('$u_1=h(x,t)$ (m)',fontsize=18)
 plt.subplot(212)
 plt.plot(Ls*xmk,U0s*u2,'-',lw=2)
 plt.xlabel('$x$ (m)',fontsize=18)
 plt.ylabel('$u_2=u(x,t)$ (m/s)',fontsize=18)


Nt = 40
dtmeet = Tend/Nt
tmeet = dtmeet
ntt = 1
dt = CFL*dxx/np.amax(c00)
print('CFL,dxx,c0',CFL,dxx,c0)
plt.pause(0.01)

U1l[1:Nkel+1] = u1[0:Nkel] 
U1l[0] = u1[Nkel-1] 
U1r[0:Nkel] = u1[0:Nkel] 
U1r[Nkel] = u1[0]   
#
# Time loop
#
tic = time.time()
while tijd <= 1.0*Tend:
    if Neqn==0:
     dt = CFL*dxx/np.amax(c00)
    else:
     dt = CFL*dxx/np.amax(u2**2/u1+np.sqrt(gtilde*u1))
    tijd = tijd+dt
    if tijd>=tmeet:
     dt = tmeet-tijd+dt+1.e-10 # tmeet minus old time plus a bit
     tijd = tmeet+1.e-10
    #       
    # Boundary conditions  
    #  
    if Nbc==0: # Nkel cells for u1, u2 and Nkel+1 cell edges for U1l, U1r, U2l, U2r periodic
     U1l[1:Nkel+1] = u1[0:Nkel] 
     U1l[0] = u1[Nkel-1]  
     U1r[0:Nkel] = u1[0:Nkel] 
     U1r[Nkel] = u1[0] 
     U2l[1:Nkel+1] = u2[0:Nkel] 
     U2l[0] = u2[Nkel-1] 
     U2r[0:Nkel] = u2[0:Nkel] 
     U2r[Nkel] = u2[0]
    elif Nbc==1: # solid walls
     U1l[1:Nkel+1] = u1[0:Nkel] 
     U1l[0] = u1[0]  
     U1r[0:Nkel] = u1[0:Nkel] 
     U1r[Nkel] = u1[Nkel-1] 
     U2l[1:Nkel+1] = u2[0:Nkel] 
     U2l[0] = -u2[0] 
     U2r[0:Nkel] = u2[0:Nkel] 
     U2r[Nkel] = -u2[Nkel-1]
     # print('u1',u1)
     # print('u2',u2)
     # print('U1l',U1l)
     # print('U2l',U2l)
     # print('U1r',U1r)
     # print('U2r',U2r)
    elif Nbc==2: # specified exact solution at x=0, x=Ld
     if Neqn==0:
      u10 = ck*np.cos(kw*0.0-omeg*tijd)
      u20 = H0*(omeg/kw)*ck*np.cos(kw*0.0-omeg*tijd)
      u1L = ck*np.cos(kw*Ld-omeg*tijd)
      u2L = H0*(omeg/kw)*ck*np.cos(kw*Ld-omeg*tijd)
     else: # h=H0+eta
      u10 = H0+ck*np.cos(kw*0.0-omeg*tijd)
      u20 = H0*(omeg/kw)*ck*np.cos(kw*0.0-omeg*tijd)
      u1L = H0+ck*np.cos(kw*Ld-omeg*tijd)
      u2L = H0*(omeg/kw)*ck*np.cos(kw*Ld-omeg*tijd)
     #
     U1l[1:Nkel+1] = u1[0:Nkel] 
     U1l[0] = u10
     U1r[0:Nkel] = u1[0:Nkel] 
     U1r[Nkel] = u1L
     # U1r[Nkel] = u1[Nkel-1]
     U2l[1:Nkel+1] = u2[0:Nkel] 
     U2l[0] = u20
     U2r[0:Nkel] = u2[0:Nkel] 
     U2r[Nkel] = u2L
    elif Nbc==3: # wave input left and extrapolating right
     u10 = Ad*np.sin(-omeg*tijd) # eta
     U1l[1:Nkel+1] = u1[0:Nkel] 
     U1l[0] = u10
     U1r[0:Nkel] = u1[0:Nkel] 
     U1r[Nkel] = u1[Nkel-1]
     u20 = (gtilde*Ad*kw/omeg)*np.sin(-omeg*tijd) # u
     U2l[1:Nkel+1] = u2[0:Nkel] 
     U2l[0] = u20
     U2r[0:Nkel] = u2[0:Nkel] 
     U2r[Nkel] = u2[Nkel-1]
     
     
     
     # U2r[Nkel] = u2[Nkel-1]
    #  
    # Define flux function, either F1=Feta, F2=FH0mu or a vector flux function
    #  
    # Note: not done elegantly should define matrix and use function
    #
    if Neqn==0: # Linear swe
     U1star = 0.5*(Hx*(U2l-U2r)/c00+U1l+U1r) #  Definition of Riemann state u1=U1star at cell edge eta
     U2star = 0.5*(U2l+U2r+c00*(U1l-U1r))    #  Definition of Riemann state u2=U2star at cell edge u
     F1 = Hx*U2star       # flux F1 = F1(U1star,U2star) = U2star = H(x)*u
     F2 = gtilde*U1star    # flux F2 = F2(U1star,U2star) = g*U1star = g*eta
     # print('F1',F1)
     # print('F2',F2)
     # plt.pause(2000)
     if Nimpl==1:
       F1 = Hx*(thetaa*U2r+(1-thetaa)*U2l) # previous F1 overwritten H*(x)*u
     if Nbc==1:
       F1[0] = 0.0
       F1[Nkel] = 0.0
    else: # Nonlinear swe 
     # 
     F1l = U2l #  h*u
     F1r = U2r #  h*u
     F2l = U2l**2/U1l+0.5*gtilde*U1l**2 # hu**2/h+0.5*g*h**2
     F2r = U2r**2/U1r+0.5*gtilde*U1r**2 # hu**2/h+0.5*g*h**2
     Sl = np.minimum(U2r/U1r-np.sqrt(gtilde*U1r),U2l/U1l-np.sqrt(gtilde*U1l))
     Sr = np.maximum(U2r/U1r+np.sqrt(gtilde*U1r),U2l/U1l+np.sqrt(gtilde*U1l))
     SrmSl = Sr-Sl
     for jj in range(0,Nkel+1):  # Not sure how to do this in vector form
      if SrmSl[jj] == 0.0: # or maybe when very small too; check?
       SrmSl[jj] = 1.0
     # flux F1 = F1(U1star,U2star) = U2star = h*u (only problem is when Sr-Sl=0):
     F1 = F1l*0.5*(1.0+np.sign(Sl))+((Sr*F1l-Sl*F1r+Sl*Sr*(U1r-U1l))/SrmSl)*0.5*(1.0+np.sign(Sr))*0.5*(1.0+np.sign(-Sl))+F1r*0.5*(1.0+np.sign(-Sr))
     # flux F2 = F2(U1star,U2star) = h*u^2+0.5*g*h^2 (only problem is when Sr-Sl=0)::
     F2 = F2l*0.5*(1.0+np.sign(Sl))+((Sr*F2l-Sl*F2r+Sl*Sr*(U2r-U2l))/SrmSl)*0.5*(1.0+np.sign(Sr))*0.5*(1.0+np.sign(-Sl))+F2r*0.5*(1.0+np.sign(-Sr))
     #   
    #  Update mean vector (u1,u2)
    #  
    #  Godunov method time stepping; same for linear and nonlinear swe's since differences in flux!
    #  
    u1[0:Nkel] = u1[0:Nkel]-(dt/dxx)*(F1[1:Nkel+1]-F1[0:Nkel]) #  DONT UNDERSTAND why one longer
    if Nimpl==1: # F2 overwritten after new u1 known
     if Nbc==0: # Nkel cells for u1, u2 and Nkel+1 cell edges for U1l, U1r, U2l, U2r periodic
      U1l[1:Nkel+1] = u1[0:Nkel] 
      U1l[0] = u1[Nkel-1]  
      U1r[0:Nkel] = u1[0:Nkel] 
      U1r[Nkel] = u1[0] 
     elif Nbc==1: # solid walls
      U1l[1:Nkel+1] = u1[0:Nkel] 
      U1l[0] = u1[0]  
      U1r[0:Nkel] = u1[0:Nkel] 
      U1r[Nkel] = u1[Nkel-1]
     elif Nbc==3:
      u10 = Ad*np.sin(-omeg*tijd) # eta
      U1l[1:Nkel+1] = u1[0:Nkel] 
      U1l[0] = u10
      U1r[0:Nkel] = u1[0:Nkel] 
      U1r[Nkel] = u1[Nkel-1]
     F2 = gtilde*((1-thetaa)*U1r+thetaa*U1l)
    #
    u2[0:Nkel] = u2[0:Nkel]-(dt/dxx)*(F2[1:Nkel+1]-F2[0:Nkel]) #  DONT UNDERSTAND why one longer
    #  
    #  Measurements for plotting
    #  
    if tijd >= tmeet:
        ntt = ntt+1
        plt.ion()
        print("dt ntt tijd Tend ",dt, ntt,tijd, Tend)
        tmeet = tmeet + dtmeet
        plt.figure(1)
        #  plt.close("all")
        plt.subplot(211)
        if Neqn==0: 
         plt.plot(xmk,H0+u1,'-',lw=2)
        else:
         plt.plot(xmk,u1,'-',lw=2)
        plt.xlabel('$x$',fontsize=18)
        plt.ylabel('$u_1=h(x,t)$',fontsize=18)
        #   plt.text(0.9, 1.05, 't=')
        #   plt.text(1.1, 1.05, tijd)
        plt.subplot(212)
        plt.plot(xmk,u2,'-',lw=2)
        plt.xlabel('$x$',fontsize=18)
        plt.ylabel('$u_2=u(x,t)$',fontsize=18)

        if Neqn==0 and Nbc==3: # linear swe and beach
         plt.figure(2)
         plt.subplot(211)
         plt.plot(Ls*xmk,H0s*(H0+u1),'-',lw=2) # eta
         plt.plot(Ls*Xce,H0s*(H0-Hx),'k',lw=3) # eta
         plt.plot(Ls*Xce,H0s*(H0+Ad*(H0/Hx)**0.25),'--r',lw=2) # eta 
         plt.xlabel('$x$ (m)',fontsize=18)
         plt.ylabel('$u_1=h(x,t)$ (m)',fontsize=18)
         plt.subplot(212)
         plt.plot(Ls*xmk,U0s*u2,'-',lw=2)
         plt.xlabel('$x$ (m)',fontsize=18)
         plt.ylabel('$u_2=u(x,t)$ (m/s)',fontsize=18)
        
        # plt.pause(1)
    #
#
# End time loop   
#
toc = time.time() - tic
print('Elapsed time (min):', toc/60)
print("Finished program!")
plt.show(block=True)
plt.pause(0.001)
plt.gcf().clear()
plt.show(block=False)



