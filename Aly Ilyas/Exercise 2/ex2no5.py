# Solve the Linear Shallow Water Equations d_t u + d_x f(u) = 0 in [x0,x1] using FEM-DG0,
# where u = [eta, H0*u]^T = [u1, u2]^T, f(u) = [H0*u, c0^2*eta]^T = [u1, c0^2*u1]^T.
# 
# Initial conditions:
# 	(a) Riemann problem
#     	u1,0(x) = u1l,  x < x_mid; u1r,  x >=x_mid
#		u2,0(x) = u2l,  x < x_mid; u2r,  x >=x_mid
# 	(b) Standing wave problem
# Boundary conditions:
#	(a) open boundary conditions (extrapolation)
#     (b) wall boundary conditions (u^l = -u^r, eta^l = eta_r)
#
# Numerical flux can be switched between the Godunov flux (a,b) and an alternating flux (b). 
#
# Visualisation of results:
# - Discrete exact solutions are plotted at grid points using matplotlab,
# - Numerical solutions are plotted using Firedarke built-in plotting function.
# - Total energy of the system is monitored.
#
# Last update: Nov 19, 2024
#
# OB: Please modify this code such that the variables used in the numerical scheme are u=[eta,u]^T with f(u) = [H*u,g*h]^T
# since that was asked and since or in that it extends to the case with H(x).
# Note that in the Riemann problem it is more convient (due to symmetry) to use [eta, H0*u]^T as variables.

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

g = 1.0
H0 = 4.0
c0 = np.sqrt(g*H0)

t0 = 0
Tend = 0.5
CFL = 1 # in [0,1]
dt = CFL*(mesh_size/c0) # time step
print('Δx = ', mesh_size, 'Δt = ', dt)
print('c0 = ', c0)
dtc = Constant(dt)

time = []
energy = []

# === define function space and Firedrake Functions ===
V = FunctionSpace(mesh, "DG", 0)
x = SpatialCoordinate(mesh)

u1_ = Function(V) # previous time-step solution, eta^{n}   OB: I would call this u1n
u1 = Function(V)  # current time-step solution, eta^{n+1}  OB: I would call this u1np1 or u1new
u1_trial = TrialFunction(V)
w1 = TestFunction(V)

u2_ = Function(V) # previous time-step solution, u^{n} OB: same see above
u2 = Function(V)  # current time-step solution, u^{n+1} OB: same see above
u2_trial = TrialFunction(V)
w2 = TestFunction(V)

# === set initial conditions and numerical flux ===
IC = "b"
Alter_flux = True # True (alternating flux) / False (Godunov flux)

if IC == "a": 
	# Riemann Problem: OB: with open extrapolating boundary
	u10_l = 0
	u10_r = 2
	u20_l = 1
	u20_r = -1
	x_mid=(x0+x1)*0.5-1e-8 # shift a bit to avoid the ambiguity at the discontinous point
	u10 = conditional(lt(x[0],x_mid), u10_l, u10_r) # UFL expression for Firedrake function u1
	u20 = conditional(lt(x[0],x_mid), u20_l, u20_r) # UFL expression for Firedrake function u2
elif IC == "b": 
	# Standing wave solution using the form: with closed or wall boundaries
	# 	eta = sin(w_k*t)*cos(k*x)
	# 	k = n*pi/L=2*pi (n=2L \in Z)
	# 	w_k = k*c0
	u10 = sin(2*pi*c0*0)*cos(2*pi*x[0])
	u20 = -c0*cos(2*pi*c0*0)*sin(2*pi*x[0]) # OB: This needs adapting when [eta,u]^T are used as variables
	
u1_.interpolate(u10)
u2_.interpolate(u20)

# plot IC with Firedrake's built-in plotting function "plot" for 1D Firedrake Function
# https://www.firedrakeproject.org/firedrake.pyplot.html#firedrake.pyplot.plot
fig1, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 9), layout='constrained')
line_10, = plot(u1_, axes=ax1, label=r'$\eta_0(x)$')   # plot DG0 FE approximation for u1
line_20, = plot(u2_, axes=ax2, label=r'$H_0 u_0(x)$')      # plot DG0 FE approximation for u2
line_10.set_color('black')
line_20.set_color('black')

#breakpoint()
#ax.scatter(x_k, u0_k, color='red', marker='.', label=r'$u_{ex},t=0$')  # plot exact solution at grid points

# === construct the linear solver for r1 and r2 ===
# It seems for 1D mesh in Firedrake, ^l/^r can be directly replaced by '+'/'-' when formulating the weak forms

# Godunov flux for (a) and (b)
# OB: may need adapting when [eta,u]^T are used as variables
F1_int = 0.5*( (u2_('+')+u2_('-')) + c0*(u1_('+')-u1_('-')) )    # Godunov flux at the interior facets
if IC=='a':
	F1_l =  0.5*( (u2_+u2_) + c0*(u1_-u1_) )    # flux at the left boundary OB: open/extrapolating boundary
	F1_r =  0.5*( (u2_+u2_) + c0*(u1_-u1_) )    # flux at the right boundary OB: open/extrapolating boundary
elif IC=='b':
	F1_l = 0.5*( ((-u2_)+u2_) + c0*(u1_-u1_) ) # flux at the right boundary OB: closed/wall boundaries
	F1_r = 0.5*( (u2_+(-u2_)) + c0*(u1_-u1_) ) # flux at the left boundary OB: closed/wall boundaries; note that minus always seems to be the inside value

F2_int = 0.5*c0*( (u2_('+')-u2_('-')) + c0*(u1_('+')+u1_('-')) )  # Godunov flux at the interior facets
if IC=='a':
	F2_l = 0.5*c0*( (u2_-u2_) + c0*(u1_+u1_) )   # flux at the left boundary OB: open/extrapolating boundary
	F2_r = 0.5*c0*( (u2_-u2_) + c0*(u1_+u1_) )   # flux at the right boundary OB: open/extrapolating boundary
elif IC == 'b':
	F2_l = 0.5*c0*( ((-u2_)-u2_) + c0*(u1_+u1_) ) # flux at the right boundary OB: closed/wall boundaries
	F2_r = 0.5*c0*( (u2_-(-u2_)) + c0*(u1_+u1_) ) # flux at the left boundary OB: closed/wall boundaries; note that minus always seems to be the inside value

# alternating flux for (b) OB: Note that this overrides the Riemann case 'a'
if IC == 'b' and Alter_flux==True:
	theta = 0.5
	F1_int = theta*u2_('-') + (1-theta)*u2_('+')
	F1_l = theta*(u2_) + (1-theta)*(-u2_)
	F1_r = theta*(-u2_) + (1-theta)*(u2_)
	F2_int = (1-theta)*c0**2*u1('-') + theta*c0**2*u1('+') # note that u1 is used instead of u1_
	F2_l = (1-theta)*c0**2*u1 + theta*c0**2*u1
	F2_r = (1-theta)*c0**2*u1 + theta*c0**2*u1

# OB Step2 for u1: weak form; note that w1=1_K is piecewise constant 1 only in its element, so has compact support
au1 = w1 * u1_trial * dx 
Lu1 = (w1 * u1_ * dx + 
      dtc * (  w1 * F1_l * ds(1)    
             - w1 * F1_r * ds(2)
             - (w1('+') - w1('-')) * F1_int * dS ))

prob_u1 = LinearVariationalProblem(au1, Lu1, u1) # OB Step 2 for u1 completed
solver_u1 = LinearVariationalSolver(prob_u1) # OB Step 4: Solver for u1

# OB: Step2 for u1: weak form; note that w1=1_K is piecewise constant 1 only in its element, so has compact support
au2 = w2 * u2_trial * dx 
Lu2 = (w2 * u2_ * dx + 
      dtc * (  w2 * F2_l * ds(1)    
             - w2 * F2_r * ds(2)
             - (w2('+') - w2('-')) * F2_int * dS ))

prob_u2 = LinearVariationalProblem(au2, Lu2, u2) # OB: Step 2 for u2 completed
solver_u2 = LinearVariationalSolver(prob_u2) # OB: Step 4: Solver for u2

t = t0 # start time
step = 0
t_output = 0.2 # < Tend
output_step = int(t_output/dt)

time.append(t0)
E0 = (0.5/H0)*assemble(u2_**2*dx) + 0.5*assemble(u1_**2*dx)
energy.append(E0)
# === time marching === OB: time loop starts
while t < Tend-0.5*dt:
		
	solver_u1.solve() # OB Call solver-1
	solver_u2.solve() # OB Call solver-2
	u1_.assign(u1)
	u2_.assign(u2)

	step += 1
	t += dt
	
	time.append(t)
	Et = (0.5/H0)*assemble(u2_**2*dx) + 0.5*assemble(u1_**2*dx) # OB: needs changing for [eta,u]^T
	energy.append(Et)

	# check intermediate result at t=t_plot
	if step == output_step:
		print("t =",t)
		# exact solution
		if IC=="a":
			conditions = [x_k < x_mid-c0*t, x_k >= x_mid+c0*t]
			values_u1 = [u10_l, u10_r]
			u1t_k = np.select( conditions, values_u1, (c0*(u10_l+u10_r)+(u20_l-u20_r))/(2.0*c0) )
			values_u2 = [u20_l, u20_r]
			u2t_k = np.select( conditions, values_u2, (c0*(u10_l-u10_r)+(u20_l+u20_r))*0.5 )	
		elif IC=="b":
			u1t_k =     np.sin(2*np.pi*c0*t) * np.cos(2*np.pi*x_k)
			u2t_k = -c0*np.cos(2*np.pi*c0*t) * np.sin(2*np.pi*x_k)
		
		ax1.scatter(x_k, u1t_k, marker='x', label=fr'$\eta_{{ex}}(x,t={t:.3f})$')
		ax2.scatter(x_k, u2t_k, marker='x', label=fr'$H_0 u_{{ex}}(x,t={t:.3f})$')
		
		# numerical solution
		plot(u1, axes=ax1, label=fr'$\eta_{{num}}(x,t={t:.3f})$')
		plot(u2, axes=ax2, label=fr'$H_0 u_{{num}}(x,t={t:.3f})$')
		
ax2.set_xlabel(r'$x$')
ax1.set_ylabel(r'$\eta(x,t)$')
ax2.set_ylabel(r'$H_0 u(x,t)$')
ax1.grid(True)
ax2.grid(True)
ax1.set_axisbelow(True)
ax2.set_axisbelow(True)
ax1.legend()
ax2.legend()

# energy plot
fig2, ax = plt.subplots(figsize=(8, 6), layout='constrained')
ax.plot(time,energy,'g-')
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$E(t)$')
ax.grid()
plt.show()