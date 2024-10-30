# Solve the linear advection equation d_t u + a d_x u = 0 (a>0) in [x0,x1] using FEM-DG0
# Initial conditions:
# 	(a) Riemann problem u0(x) = ul,  x < x_mid; ur,  x >=x_mid
# 	(b) traveling wave u0(x) = sin(2*pi*x/Lx)
# Boundary condition: specify upwind boundary value u(x0,t) for t>0
# Discrete exact solutions are plotted at grid points using matplotlab,
# Numerical solutions are plotted using Firedarke built-in plotting function.

from firedrake import *
import matplotlib.pyplot as plt
from firedrake.pyplot import plot
import numpy as np

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
t0 = 0
Tend = 0.5
CFL = 1.0 # in [0,1]
dt = CFL*(mesh_size/a) # time step
dtc = Constant(dt)

# === define function space and Firedrake Functions ===
V = FunctionSpace(mesh, "DG", 0)
x = SpatialCoordinate(mesh)
n = FacetNormal(mesh)

u_ = Function(V) # previous time-step solution
u = Function(V)  # current time-step solution
u_trial = TrialFunction(V)
w = TestFunction(V)

# === set initial condition ===
IC = "a"

if IC == "a":
	u0_l=5
	u0_r=2
	x_mid=(x0+x1)*0.5-1e-8 # shift a bit to avoid the ambiguity at the discontinous point
	u0_k = np.where(x_k < x_mid, u0_l, u0_r)     # numpy array
	u0 = conditional(lt(x[0],x_mid), u0_l, u0_r) # UFL expression
	u_in = Constant(u0_l)   # inflow boundary condition
elif IC == "b":
	u0_k = np.sin(2*np.pi*x_k/Lx)  # numpy array
	u0 = sin(2*pi*x[0]/Lx)         # UFL expression
	u_x0 = np.sin(2*np.pi*x0/Lx-(2*np.pi*a/Lx)*(t0+dt))
	u_in = Constant(u_x0)   # inflow boundary condition

u_.interpolate(u0)

# plot IC with Firedrake's built-in plotting function "plot" for 1D Firedrake Function
# https://www.firedrakeproject.org/firedrake.pyplot.html#firedrake.pyplot.plot
fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
line0, = plot(u_, axes=ax, label=r'$u_0(x)$')   # plot DG0 FE approximation
line0.set_color('black')
#ax.scatter(x_k, u0_k, color='red', marker='.', label=r'$u_{ex},t=0$')  # plot exact solution at grid points

# === construct the linear solver ===
F_in  = a * u_in     # flux at the left boundary, inflow BC
F_out = a * u_       # flux at the right boundary, outflow/open BC
F_int = a * u_('+')  # Godunov flux at the interior facets

au = w * u_trial * dx
Lu = (w * u_ * dx + 
      dtc * (  w * F_in * ds(1)    # inflow boundary integral, ds(1) stands for the left boundary
             - w * F_out * ds(2)    # outflow boundary integral, ds(2) stands for the right boundary
             - (w('+') - w('-')) * F_int * dS ))   # Godunov flux used for the interior facets

problem = LinearVariationalProblem(au, Lu, u)
solver = LinearVariationalSolver(problem)

t = t0 # start time
step = 0
t_output = 0.3 # < Tend
output_step = int(t_output/dt)

# === time marching ===
while t < Tend-0.5*dt:
	# update time-dependent boundary condition
	if IC=="b":
		u_in.assign(sin(2*pi*x0/Lx-(2*pi*a/Lx)*(t+dt))) 
		
	solver.solve()
	u_.assign(u)

	step += 1
	t += dt
	
	# check intermediate result at t=t_plot
	if step == output_step:
		print("t=",t)
		if IC=="a":
			ut_k = np.where(x_k-a*t < x_mid, u0_l, u0_r)
		elif IC=="b":
			ut_k = np.sin(2*np.pi*x_k/Lx-(2*np.pi*a/Lx)*t)
		ax.scatter(x_k, ut_k, marker='x', label=fr'$u_{{ex}}(x,t={t:.3f})$') 
		plot(u, axes=ax, label=fr'$u_{{num}}(x,t={t:.3f})$')

ax.set_title(r'DG0 Finite Element Solution of the Linear Advection Equation $\partial_t u + a \partial_x u = 0 \, (a > 0)$')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$u(x,t)$')
ax.grid(True)
ax.set_axisbelow(True)
ax.legend()          
plt.show()

