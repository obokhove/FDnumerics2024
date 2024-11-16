# Modified to solve Equation (1) from exercises list 2 using FEM-DG0
# Initial conditions:
# 	(a) Riemann problem u0(x) = ul,  x < x_mid; ur,  x >=x_mid -> tested
# 	(b) traveling wave u0(x) = sin(2*pi*x/Lx) -> not testes
# Boundary condition: specify upwind boundary value u(x0,t) for t>0
# Visualisation of results:
# - Discrete exact solutions are plotted at grid points using matplotlab,
# - Numerical solutions are plotted using Firedarke built-in plotting function.
#%%
from firedrake import *
import matplotlib.pyplot as plt
from firedrake.pyplot import plot
import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = "2"
#                              E_k
#          x0                |<--->|                       x1
#       ---|-----|-----|-----|-----|-----|-----|-----|-----|---> x
#         x_0   x_1   ...  x_k-1  x_k  x_k+1  ...  x_Nx-1 x_Nx   <= grid points  

# === discretisation ===
Lx = 1 # length of the interval
x0 = 0 # x in [x0,x1]
x1 = x0+Lx
Nx = 100 # number of elements
mesh_size = Lx/Nx # uniform grid, DO NOT USE dx!!!

x_k=np.linspace(x0,x1,num=Nx+1) # grid points, k=0,1,...,Nx, for point-wise plot of exact solutions
#https://www.firedrakeproject.org/firedrake.html#firedrake.utility_meshes.IntervalMesh
mesh = IntervalMesh(Nx,Lx)

a = 1.0
H0 =1.0
g = 1
c0 = np.sqrt(H0*g)
t0 = 0
Tend = 0.5
CFL = 1 # in [0,1]
dt = CFL*(mesh_size/np.sqrt(H0*g)) # time step
dtc = Constant(dt)
dxc = Constant(mesh_size)
print(dtc)
# === define function space and Firedrake Functions ===
V = FunctionSpace(mesh, "DG", 0)
x = SpatialCoordinate(mesh)
nx = FacetNormal(mesh)[0]
an = 0.5*(a * nx + abs(a * nx))

u_ = Function(V) # previous time-step solution
u = Function(V)  # current time-step solution
u_trial = TrialFunction(V)
w = TestFunction(V)

eta_ =  Function(V) # previous height solution
eta = Function(V)  # current time-step solution
eta__trial = TrialFunction(V)
eta__test = TestFunction(V)


# === set initial condition ===
IC = "a"

if IC == "a":
	u0_l=0.5
	u0_r=0
	x_mid=(x0+x1)*0.5-1e-8 # shift a bit to avoid the ambiguity at the discontinous point
	u0_k = np.where(x_k < x_mid, u0_l, u0_r)     # numpy array
	u0 = conditional(lt(x[0],x_mid), u0_l, u0_r) # UFL expression
	u_in = Constant(-u0_l)   # inflow boundary condition
	eta0_l=1
	eta0_r=0
	eta0_k = np.where(x_k < x_mid, eta0_l, eta0_r)     # numpy array
	eta0 = conditional(lt(x[0],x_mid), eta0_l, eta0_r) # UFL expression
	eta_in = Constant(eta0_l)   # inflow boundary condition 
elif IC == "b":
	u0_k = np.sin(2*np.pi*x_k/Lx)  # numpy array
	u0 = sin(2*pi*x[0]/Lx)         # UFL expression
	u_x0 = np.sin(2*np.pi*x0/Lx-(2*np.pi*g/Lx)*(t0+dt))
	u_in = Constant(u_x0)   # inflow boundary condition
	eta0_k = np.sin(2*np.pi*x_k/Lx)  # numpy array
	eta0 = sin(2*pi*x[0]/Lx)         # UFL expression
	eta_x0 = np.sin(2*np.pi*x0/Lx-(2*np.pi*H0/Lx)*(t0+dt))
	eta_in = Constant(eta_x0)

u_.interpolate(u0)
eta_.interpolate(eta0)

###########################################
### PLOT IC ###
# plot IC with Firedrake's built-in plotting function "plot" for 1D Firedrake Function
# https://www.firedrakeproject.org/firedrake.pyplot.html#firedrake.pyplot.plot
fig, ax = plt.subplots(2, figsize=(8, 5), layout='constrained')
# line0, = plot(u_, axes=ax[0], label=r'$u_0(x)$')   # plot DG0 FE approximation
# line0.set_color('black')
# line1, = plot(eta_, axes=ax[1], label=r'$eta_0(x)$')   # plot DG0 FE approximation
# line1.set_color('red')
###########################################
# print(len(x_k))
eta12 = np.array([eta_.at(x) for x in x_k]) # eta12 = np.array([eta0.at(x) for x in xvals])
ax[0].plot(x_k, eta12, label=r'$eta_0(x)$')
phi12 = np.array([u_.at(x) for x in x_k])
ax[1].plot(x_k, phi12,  label=r'$u_0(x)$')
# ax1.plot(Ls*xvals,H0s*(H0+eta12), label = f't = {t:.3f}')


#breakpoint()
# #ax.scatter(x_k, u0_k, color='red', marker='.', label=r'$u_{ex},t=0$')  # plot exact solution at grid points
print(eta_.dat.data[0])
# print(len(u_))
###### FLUXES #####
# === construct the linear solver ===
# ##bounduary
F_1in  = H0*u_ #(H0*(u_in + u_.dat.data[0]) + c0*(eta_in - eta_.dat.data[0]))/2.0     # flux at the left boundary, inflow BC
F_1out = H0*u_       # flux at the right boundary, outflow/open BC

F_2in  = g*eta_    # flux at the left boundary, inflow BC
F_2out = g*eta_
# #F_int = a * u_('+')  # Godunov flux at the interior facets, turns out to be the simplified form

# ### internal
# F1_int = (H0*(an('-')*u_('-') +an('+')* u_('+')) - c0*((an('-')*eta_('-') - an('+')*eta_('+'))))/2.0
# F2_int = (H0*(u_('+') - u_('-')) + c0*((eta_('-') + eta_('+'))))*g/(2.0*c0)

F1_int = (H0*(u_('-') + u_('+')) - c0*((eta_('-') - eta_('+'))))/2.0
F2_int = (H0*(u_('+') - u_('-')) + c0*((eta_('-') + eta_('+'))))*g/(2.0*c0)


aeta = eta__test*eta__trial*dx
Leta =  (eta__test * eta_ * dx + 
      dtc * (  eta__test * F_1in * ds(1)    # inflow boundary integral, ds(1) stands for the left boundary
             - eta__test * F_1out * ds(2)    # outflow boundary integral, ds(2) stands for the right boundary
             - (eta__test('+') - eta__test('-')) * F1_int * dS ))

au = w * u_trial * dx
Lu = (w * u_ * dx + 
      dtc * (  w * F_2in * ds(1)    # inflow boundary integral, ds(1) stands for the left boundary
             - w * F_2out * ds(2)    # outflow boundary integral, ds(2) stands for the right boundary
             - (w('+') - w('-')) * F2_int * dS ))   # Godunov flux used for the interior facets

problem1 = LinearVariationalProblem(aeta, Leta, eta)
solver1 = LinearVariationalSolver(problem1)


problem2 = LinearVariationalProblem(au, Lu, u)
solver2 = LinearVariationalSolver(problem2)


def u_analitcal (x,t,x_mid,ul,ur,etal,etar, H0, c0):
    if x < x_mid -c0*t:
        return ul
    elif x < x_mid +  c0*t and x> x_mid -c0*t:
        return (H0*(ul + ur) + c0*(etal - etar))/(2.0*H0)
    elif x > x_mid + c0*t:
        return ur
    
def eta_analitcal (x,t,x_mid,ul,ur,etal,etar, H0, c0):
    if x < x_mid -c0*t:
        return etal
    elif x < x_mid +  c0*t and x> x_mid -c0*t:
        return (H0*(ul - ur)/c0 + (etal + etar))/(2.0)
    elif x > x_mid + c0*t:
        return etar

t = t0 # start time
step = 0
# t_output = 0.3 # < Tend
# output_step = int(t_output/dt)

# === time marching ===
while t < 10*Tend-0.5*dt:
	# update time-dependent boundary condition
	if IC=="b":
		u_in.assign(sin(2*pi*x0/Lx-(2*pi*g/Lx)*(t+dt))) 
		eta_in.assign(sin(2*pi*x0/Lx-(2*pi*H0/Lx)*(t+dt))) 
		
	solver1.solve()
	solver2.solve()

	u_.assign(u)
	eta_.assign(eta)

	step += 1
	t += dt
	
	t_output = 0.3 # < Tend
	output_step = int(t_output/dt)
	# check intermediate result at t=t_plot
	if step == output_step:
		print("t=",t)
		if IC=="a":
			ut_k = np.where(x_k-a*t < x_mid, u0_l, u0_r)
			etat_k = np.where(x_k-a*t < x_mid, eta0_l, eta0_r)
		elif IC=="b":
			ut_k = np.sin(2*np.pi*x_k/Lx-(2*np.pi*a/Lx)*t)
		
  		# u_real_sol = [u_analitcal (x,0.5,u0_l,u0_r,eta0_l,eta0_r, H0, c0)]
		eta12 = np.array([eta_.at(x) for x in x_k]) 
		phi12 = np.array([u_.at(x) for x in x_k])
		eta_real_sol = [eta_analitcal (x,t,x_mid,u0_l,u0_r,eta0_l,eta0_r, H0, c0) for x in x_k]
		ax[0].plot(x_k, eta_real_sol, label=fr'$\eta_{{analitical}}(x,t={t:.3f})$')
		ax[0].scatter(x_k, eta12, label=fr'$eta_{{num}}(x,t={t:.3f})$',c = 'orange', marker = 'x')
		u_real_sol = [u_analitcal (x,t,x_mid,u0_l,u0_r,eta0_l,eta0_r, H0, c0) for x in x_k]
		ax[1].plot(x_k, u_real_sol, label=fr'$u_{{analitical}}(x,t={t:.3f})$')
		ax[1].scatter(x_k, phi12, label=fr'$u_{{num}}(x,t={t:.3f})$',c = 'orange', marker = 'x')
	t_output = 0.6 # < Tend
	output_step = int(t_output/dt)
	# check intermediate result at t=t_plot
	if step == output_step:
		print("t=",t)
		if IC=="a":
			ut_k = np.where(x_k-a*t < x_mid, u0_l, u0_r)
			etat_k = np.where(x_k-a*t < x_mid, eta0_l, eta0_r)
		elif IC=="b":
			ut_k = np.sin(2*np.pi*x_k/Lx-(2*np.pi*a/Lx)*t)
		
  		# u_real_sol = [u_analitcal (x,0.5,u0_l,u0_r,eta0_l,eta0_r, H0, c0)]
		eta12 = np.array([eta_.at(x) for x in x_k]) 
		phi12 = np.array([u_.at(x) for x in x_k])
		eta_real_sol = [eta_analitcal (x,t,x_mid,u0_l,u0_r,eta0_l,eta0_r, H0, c0) for x in x_k]
		ax[0].plot(x_k, eta_real_sol, label=fr'$\eta_{{analitical}}(x,t={t:.3f})$', c = 'purple')
		ax[0].scatter(x_k, eta12, label=fr'$eta_{{num}}(x,t={t:.3f})$', c = 'purple', marker = 'x')
		u_real_sol = [u_analitcal (x,t,x_mid,u0_l,u0_r,eta0_l,eta0_r, H0, c0) for x in x_k]
		ax[1].plot(x_k, u_real_sol, label=fr'$u_{{analitical}}(x,t={t:.3f})$', c = 'purple')
		ax[1].scatter(x_k, phi12, label=fr'$u_{{num}}(x,t={t:.3f})$', c = 'purple', marker = 'x')
  

ax[1].set_xlabel(r'$x$')
ax[1].set_ylabel(r'$u(x,t)$')
ax[0].set_ylabel(r'$\eta(x,t)$')
ax[0].grid(True)
ax[1].grid(True)
# ax.set_axisbelow(True)
ax[0].legend()
ax[1].legend()          

# plt.show()
plt.savefig('Godunov_Riemann_IC.png', dpi = 500, bbox_inches = 'tight')

# %%
