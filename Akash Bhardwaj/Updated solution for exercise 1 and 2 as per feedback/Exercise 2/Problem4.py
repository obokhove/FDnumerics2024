# Solve the linear advection equation d_t u + a d_x u = 0 (a>0) in [x0,x1] using FEM-DG0
# Initial conditions:
#   (a) Riemann problem u0(x) = ul, x < x_mid; ur, x >= x_mid
#   (b) traveling wave u0(x) = sin(2*pi*x/Lx)
# Boundary condition: specify upwind boundary value u(x0,t) for t>0
# Visualisation of results:
# - Discrete exact solutions are plotted at grid points using matplotlib,
# - Numerical solutions are plotted using Firedrake's built-in plotting function.

from firedrake import *
import matplotlib.pyplot as plt
from firedrake.pyplot import plot
import numpy as np

#                              E_k
#          x0                |<--->|                       x1
#       ---|-----|-----|-----|-----|-----|-----|-----|-----|---> x
#         x_0   x_1   ...  x_k-1  x_k  x_k+1  ...  x_Nx-1 x_Nx   <= grid points  

# === discretisation ===
Lx = 2.5               # length of the interval
x0 = 0.0
x1 = x0 + Lx
Nx = 100               # number of elements
mesh_size = Lx / Nx    # uniform grid, DO NOT USE dx!

x_k = np.linspace(x0, x1, num=Nx+1)  # grid points for exact solution
mesh = IntervalMesh(Nx, Lx)

a = 1.0
t0 = 0.0

# --- CHANGED: define period T_p and run until 4*T_p
T_p = Lx / a          # if Lx=1 and a=1 => T_p=1
Tend = 4.0 * T_p      # run until 4*T_p = 4

# --- CHANGED: keep the same formula for dt but it will be smaller
CFL = 0.5            # in [0,1]
dt = CFL*(mesh_size / abs(a))  # => 1.0*(0.01/1.0)=0.01
dtc = Constant(dt)

# === define function space and Firedrake Functions ===
V = FunctionSpace(mesh, "DG", 0)
x = SpatialCoordinate(mesh)
nx = FacetNormal(mesh)[0]
an = 0.5*(a*nx + abs(a*nx))

u_ = Function(V)  # previous time-step solution
u = Function(V)   # current time-step solution
u_trial = TrialFunction(V)
w = TestFunction(V)

# === set initial condition ===
IC = "a"  # 'a': Riemann, 'b': traveling wave

if IC == "a":
    u0_l = 5
    u0_r = 2
    x_mid = (x0 + x1)*0.5 - 1e-8
    u0_k = np.where(x_k < x_mid, u0_l, u0_r)    # numpy array for exact plot
    u0 = conditional(lt(x[0], x_mid), u0_l, u0_r)  # UFL expression
    u_in = Constant(u0_l)   # inflow boundary condition
elif IC == "b":
    u0_k = np.sin(2*np.pi*x_k / Lx)
    u0 = sin(2*pi*x[0]/Lx)
    u_x0 = np.sin(2*np.pi*x0/Lx - (2*np.pi*a/Lx)*(t0+dt))
    u_in = Constant(u_x0)   # inflow boundary condition

u_.interpolate(u0)

# Plot initial condition
fig, ax = plt.subplots(figsize=(8,5), layout='constrained')
line0, = plot(u_, axes=ax, label=r'$u_0(x)$')
line0.set_color('black')

# === construct the linear solver ===
F_in  = a * u_in     # flux at the left boundary
F_out = a * u_       # flux at the right boundary (open)
F_int = an('+')*u_('+') - an('-')*u_('-')  # general Godunov flux

au = w * u_trial * dx
Lu = (w * u_ * dx +
      dtc*( w * F_in * ds(1)
           - w * F_out * ds(2)
           - (w('+') - w('-')) * F_int * dS ))

problem = LinearVariationalProblem(au, Lu, u)
solver = LinearVariationalSolver(problem)

t = t0
step = 0

# --- CHANGED: we want to plot from 3*T_p to 4*T_p => times [3,4]
# So define multiple output times if desired, e.g. at t=3.0 and t=4.0
t_output = [0.3, 0.6,0.9]
output_steps = [int(t_out / dt) for t_out in t_output]
# e.g. int(3.0/0.01)=300, int(4.0/0.01)=400

# === time marching ===
while t < Tend - 0.5*dt:
    # update time-dependent boundary if traveling wave
    if IC == "b":
        u_in.assign(sin(2*pi*x0/Lx - (2*pi*a/Lx)*(t + dt)))

    solver.solve()
    u_.assign(u)

    step += 1
    t += dt

    # only plot if step is in output_steps => t is exactly 3.0 or 4.0
    if step in output_steps:
        print(f"t= {t:.2f}")
        if IC == "a":
            # exact Riemann solution
            ut_k = np.where(x_k - a*t < x_mid, u0_l, u0_r)
        else:
            # traveling wave
            ut_k = np.sin(2*np.pi*x_k / Lx - (2*np.pi*a/Lx)*t)

        ax.scatter(x_k, ut_k, marker='x',
                   label=fr'$u_{{ex}}(x,t={t:.2f})$')
        plot(u, axes=ax, label=fr'$u_{{num}}(x,t={t:.2f})$')

ax.set_title(r'DG0 various profiles (exact and numerical) at CFL=$0.5$')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$u(x,t)$')
ax.grid(True)
ax.set_axisbelow(True)
ax.legend()
plt.show()
