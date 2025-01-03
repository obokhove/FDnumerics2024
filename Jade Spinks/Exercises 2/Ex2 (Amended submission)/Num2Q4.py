# Modified for Godonov scheme for linear-shallow wave eqs 
#Adding in lines for eta 
#
# Solve the linear advection equation d_t u + a d_x u = 0 (a>0) in [x0,x1] using FEM-DG0
# Initial conditions:
# 	(a) Riemann problem u0(x) = ul,  x < x_mid; ur,  x >=x_mid
# 	(b) traveling wave u0(x) = sin(2*pi*x/Lx)
# Boundary condition: specify upwind boundary value u(x0,t) for t>0
# Visualisation of results:
# - Discrete exact solutions are plotted at grid points using matplotlab,
# - Numerical solutions are plotted using Firedarke built-in plotting function.

from firedrake import *
import matplotlib.pyplot as plt
from firedrake.pyplot import plot
import numpy as np


#                              E_k
#          x0                |<--->|                       x1
#       ---|-----|-----|-----|-----|-----|-----|-----|-----|---> x
#         x_0   x_1   ...  x_k-1  x_k  x_k+1  ...  x_Nx-1 x_Nx   <= grid points  

#========= Mesh/solution variables ===============================================================

#Set for interval [0,1], nablax = 1/100
Lx = 1 # length of the interval
x0 = 0 # x in [x0,x1]
x1 = x0+Lx
Nx = 100 # number of elements
mesh_size = Lx/Nx # uniform grid, DO NOT USE dx!!!

x_k=np.linspace(x0,x1,num=Nx+1) # grid points, k=0,1,...,Nx, for point-wise plot of exact solutions

mesh = IntervalMesh(Nx,Lx)

t0 = 0
Tend = 0.5
cfl = 0.5 # in [0,1]
h0 = 1 #unit value for H_0 constant
g =  1 #m^2/s value for gravity
c0 = np.sqrt(h0*g) #defining sound speed
dt = cfl*(mesh_size/c0) # time step where dt = cfl*nablax / max lambda
dtc = Constant(dt)

#========= Function spaces and test/trial functions ======================================================
V = FunctionSpace(mesh, "DG", 0)
x = SpatialCoordinate(mesh)
nx = FacetNormal(mesh)[0]
#an = 0.5*(a * nx + abs(a * nx))

u_ = Function(V) # previous time-step solution t_n
u = Function(V)  # current time-step solution t_n+1
u_trial = TrialFunction(V)
u_test = TestFunction(V)

#Adding additional variable e = eta into code for shallow water equations
e_ = Function(V)
e = Function(V)
e_trial = TrialFunction(V)
e_test = TestFunction(V)

# ========= Setting Initial Conditions ========================================================================
IC = "a"

#Choosing Riemann problem or standing wave ICs
if IC == "a": 
	u0_l=9
	u0_r=3
	x_mid=((x0+x1)*0.5-1e-8) # shift a bit to avoid the ambiguity at the discontinous point
	u0_k = np.where(x_k < x_mid, u0_l, u0_r)     # numpy array
	u0 = conditional(lt(x[0],x_mid), u0_l, u0_r) # UFL expression
	u_in = Constant(-u0_l)   # inflow boundary condition on LHS of mesh, U_1^n = -U_0^n = -u_l from Qu3
	e0_l=4
	e0_r=2
	e0_k = np.where(x_k < x_mid, e0_l, e0_r) 
	e0 = conditional(lt(x[0],x_mid), e0_l, e0_r)
	e_in = Constant(e0_l) #LHS of mesh boundary condition, E_1^n = E_0^n = eta_l


u_.interpolate(u0)
e_.interpolate(e0)

#========= Plotting initial conditions =================================================================================

# plot IC with Firedrake's built-in plotting function "plot" for 1D Firedrake Function
# https://www.firedrakeproject.org/firedrake.pyplot.html#firedrake.pyplot.plot
fig, (ax_u, ax_e) = plt.subplots(2,1, figsize=(8, 6), layout='constrained')
#line_u, = plot(u_, axes=ax_u, label=r'$u_0(x)$')   # plot DG0 FE approximation 
#line_u.set_color('#264b96')
#line_e, = plot(e_, axes=ax_e, label=r'$η_0(x)$')   # plot DG0 FE approximation 
#line_e.set_color('magenta')



#ax_u.scatter(x_k, u0_k, color='red', marker='.', label=r'$u_{ex},t=0$')  # plot exact solution at grid points


#========= Defining flux functions ==========================================================================================

Fu_in  = h0 * u_    # flux at the left boundary, for u, inflow BC
Fu_out = h0 * u_       # flux at the right boundary, for u, outflow/open BC



Fe_in  = g*e_  # flux at the left boundary, for u, inflow BC
Fe_out = g*e_       # flux at the right boundary, for u, outflow/open BC


Fu_gen = (h0*(u_('-') + u_('+')) + c0*((e_('+') - e_('-'))))/2.0
Fe_gen = (h0*g*(u_('+') - u_('-')) + c0*g*((e_('+') + e_('-'))))/(2.0)


#F_gen = a * u_('+')  # Godunov flux at the interior facets, turns out to be the simplified form
#Fu_gen = an('+')*u_('+') - an('-')*u_('-') # general form
# "Note that there is no guaranteed relationship between the global x coordinate and the facet orientations.
# It can be different from one facet to the next." (--DH)

#========= Defining weak form  ==========================================================================================

# From Q_k^n+1 = Q_k^n - dt/nabla x F(Q_k^n, Q_k+1^n) + dt/nabla x F(Q_k-1^n, Q_k^n), q = (eta, u), f= (h0u, geta), multiplying by test
au = u_test * u_trial * dx
Lu = (u_test * u_ * dx) + (dt) * (  u_test * Fe_in * ds(1)    # inflow boundary integral, ds(1) stands for the left boundary
             - u_test * Fe_out * ds(2)    # outflow boundary integral, ds(2) stands for the right boundary
             - (u_test('+') - u_test('-')) * Fe_gen * dS )   # Godunov flux used for the interior facets

ae = e_test * e_trial * dx
Le = (e_test  * e_ * dx + 
 (dt) * (  e_test * Fu_in * ds(1)    # inflow boundary integral, ds(1) stands for the left boundary
             - e_test * Fu_out * ds(2)    # outflow boundary integral, ds(2) stands for the right boundary
             - (e_test('+') - e_test('-')) * Fu_gen * dS ))   # Godunov flux used for the interior facets



problem_u = LinearVariationalProblem(au, Lu, u)
solver_u = LinearVariationalSolver(problem_u)

problem_e = LinearVariationalProblem(ae, Le, e)
solver_e = LinearVariationalSolver(problem_e)

#========= Time-steps  ==========================================================================================

t = t0 # start time
step = 0
t_output = 0.05 # < Tend
output_step = int(t_output/dt)



# === time marching ===
while t < Tend-0.5*dt:
	# update time-dependent boundary condition 
	

	#u_.dat.data[:] = abs(u_.dat.data) - forces absolute values
	#e_.dat.data[:] = abs(e_.dat.data) - forces absolute values

	step += 1
	t += dt

	solver_u.solve()
	solver_e.solve()

	u_.assign(u)
	e_.assign(e)

	
	# check intermediate result at t=t_plot
	
	if step == output_step:
		print("t=",t)
		if IC=="a":
			ut_k = np.where(x_k<(x_mid-c0*t), u0_l, np.where(x_k>(x_mid + c0*t),u0_r, (0.5*h0*(u0_r+u0_l) + 0.5*c0*(e0_l-e0_r)) ))
			et_k = np.where(x_k< (x_mid-c0*t), e0_l, np.where(x_k>(x_mid + c0*t),e0_r, (h0*g*0.5*(u0_l-u0_r) + 0.5*g*c0*(e0_l+e0_r)) ))
		elif IC=="b":
			ut_k = np.sin(2*np.pi*x_k/Lx-(2*np.pi*a/Lx)*t)
		
		uvec = u_.dat.data[:] 
		evec = e_.dat.data[:]
		xtest = x_k[:100] #for display only - x_k and uvec are not the same length for whatever reason
		ax_u.scatter(xtest, uvec, c='#5E49D0', marker=".", label=fr'$u_{{num}}(x,t={t:.3f})$')
		ax_e.scatter(xtest, evec, c='#E50101', marker='.', label=fr'$η_{{num}}(x,t={t:.3f})$')
		ax_u.plot(x_k, ut_k, color='#4DAE57', label=fr'$u_{{ex}}(x,t={t:.3f})$') 
		ax_e.plot(x_k, et_k, color='#D672CC', label=fr'$η_{{ex}}(x,t={t:.3f})$') 
		
ax_u.set_title(f'DG0 Solutions (u) - t_output={t_output:.3f}, CFL={cfl:.2f}')
ax_e.set_title(f'DG0 Solutions (η) - t_output={t_output:.3f}, CFL={cfl:.2f}')
ax_u.set_xlabel(r'$x$')
ax_u.set_ylabel(r'$u(x,t)$')
ax_u.grid(True)
ax_u.set_axisbelow(True)
ax_u.legend()    

ax_e.set_xlabel(r'$x$')
ax_e.set_ylabel(r'$η(x,t)$')
ax_e.grid(True)
ax_e.set_axisbelow(True)
ax_e.legend()  

plt.show()
plt.savefig(f'Num2Q4t{t_output:.3f}cfl{cfl:.2f}.png')

#utd_k = ut_k[:100]
#diff = np.sqrt((uvec- utd_k)**2)
#print(avg(diff))



