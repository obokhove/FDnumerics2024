# Solve the linear advection equation d_t u + a d_x u = 0 (a>0) in [x0,x1] using Godunov method
# Initial conditions:
# 	(a) Riemann problem u0(x) = ul,  x < x_mid; ur,  x >=x_mid
# 	(b) traveling wave u0(x) = sin(2*pi*x/Lx)
# Boundary condition: specify upwind boundary value u(x0,t) for t>0

import numpy as np
import matplotlib.pyplot as plt

#                                 CV_k
#          x0                   |<--->|                    x1
# ---|-----|-----|-----|-----|-----|-----|-----|-----|-----|---> x
#   x_-1  x_0   x_1   ...  x_k-1  x_k  x_k+1  ...  x_Nx-1 x_Nx   <= grid points
#         u[0]                                            u[Nx]  <= Python index
#    ^
#  ghost point (a>0)

# === discretisation ===
Lx = 1 # length of the interval
x0 = 0 # x in [x0,x1]
x1 = x0+Lx
Nx = 100 # total number of CVs: Nx+1, ghost CV at x=x_{-1}
dx = Lx/Nx # uniform grid
x_k=np.linspace(x0,x1,num=Nx+1) # grid points, k=0,1,...,Nx

a = 1.0
Tend = 0.5
CFL = 1.0 # in [0,1]
dt = CFL*(dx/a) # time step

# === setting initial condition ===
IC = "a"

if IC == "a":
	u0_l=5
	u0_r=2
	x_mid=(x0+x1)*0.5-1e-8 # shift a bit to avoid the ambiguity at the discontinous point
	u0 = np.where(x_k < x_mid, u0_l, u0_r) # straightforward
	#u0 = u0_l*np.heaviside(x_mid-x_k,0) + u0_r*np.heaviside(x_k-x_mid,1) # tricky at x=x_mid, not recommended
elif IC == "b":
	u0 = np.sin(2*np.pi*x_k/Lx)

# plot IC
fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
ax.plot(x_k, u0, c='k', label=r'$u_0(x)$')


t = 0.0 # start time
u_ = u0.copy() # U_k^{n-1}
u = u0.copy()  # U_k^{n}

step = 0
output_dt = 0.1
output_freq = int(output_dt/dt)

# === time marching ===
while t < Tend-0.5*dt:
	# upwind boundary condition: specify inflow at x=x0
	if IC=="a":
		u[0]=u0_l
	elif IC=="b":
		u[0]=np.sin(2*np.pi*x0/Lx-(2*np.pi*a/Lx)*(t+dt))

	for k in range(1,Nx+1):
		# Godunov Flux when a>0
		F_r = a*u_[k]    # F(U_{k},U_{k+1})
		F_l = a*u_[k-1]  # F(U_{k-1},U_{k})
		# update solution using Godunov method
		u[k]=u_[k]- (dt/dx)*(F_r-F_l)
	step += 1
	t += dt
	u_ = u.copy()

	# plot intermediate results
	if step % output_freq == 0:
		print("t=",t)
		if IC=="a":
			#ut = u0_l*np.heaviside(x_mid-(x_k-a*t),0) + u0_r*np.heaviside((x_k-a*t)-x_mid,1) ! tricky at x=x_mid+a*t
			ut = np.where(x_k-a*t < x_mid, u0_l, u0_r)
		elif IC=="b":
			ut = np.sin(2*np.pi*x_k/Lx-(2*np.pi*a/Lx)*t)
		ax.plot(x_k, ut, label=fr'$u_{{ex}}(x,t={t:.3f})$')
		ax.plot(x_k, u, ls='--', label=fr'$u_{{num}}(x,t={t:.3f})$')

ax.set_title(r'Finite Volumn Solution of the Linear Advection Equation $\partial_t u + a \partial_x u = 0 \, (a > 0)$')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$u(x,t)$')
ax.grid()
#ax.set_title("")
ax.legend()          
plt.show()   


