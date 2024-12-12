from firedrake import *
import matplotlib.pyplot as plt
from firedrake.pyplot import plot
from firedrake import SpatialCoordinate, conditional
from ufl import sin, pi
import numpy as np
import os

# Set environment variables
os.environ["OMP_numerical_THREADS"] = "2"

# === set initial condition ===
IC = "b"  # Set to 'a' for the Riemann problem, 'b' for the standing wave

# Standing wave parameter
am = 4 # wavenumber

# === discretisation ===
Lx = 1  # length of the interval
x0 = 0  # x in [x0,x1]
x1 = x0 + Lx
Nx = 1000  # numericalber of elements
mesh_size = Lx / Nx  # uniform grid

x_k = np.linspace(x0, x1, num=Nx + 1)  # grid points for plot of exact solutions
mesh = IntervalMesh(Nx, Lx)

a = 1.0
H0 = 1.0
g = 1
c0 = np.sqrt(H0 * g)
t0 = 0
Tend = 0.5
CFL = 1  # in [0,1]
dt = CFL * (mesh_size / c0)  # time step
dtc = Constant(dt)
dxc = Constant(mesh_size)

# === define function space and Firedrake Functions ===
V = FunctionSpace(mesh, "DG", 0)
x = SpatialCoordinate(mesh)
nx = FacetNormal(mesh)[0]
an = 0.5 * (a * nx + abs(a * nx))

u_ = Function(V)  # previous time-step solution
u = Function(V)  # current time-step solution
u_trial = TrialFunction(V)
w = TestFunction(V)

eta_ = Function(V)  # previous height solution
eta = Function(V)  # current time-step solution
eta__trial = TrialFunction(V)
eta__test = TestFunction(V)

if IC == "a":
    u0_l = 0.5
    u0_r = 0
    x_mid = (x0 + x1) * 0.5 - 1e-8  # shift a bit to avoid ambiguity at the discontinuous point
    u0_k = np.where(x_k < x_mid, u0_l, u0_r)  # numericalpy array
    u0 = conditional(lt(x[0], x_mid), u0_l, u0_r)  # UFL expression
    u_in = Constant(-u0_l)  # inflow boundary condition
    eta0_l = 1
    eta0_r = 0
    eta0_k = np.where(x_k < x_mid, eta0_l, eta0_r)  # numericalpy array
    eta0 = conditional(lt(x[0], x_mid), eta0_l, eta0_r)  # UFL expression
    eta_in = Constant(eta0_l)  # inflow boundary condition
elif IC == "b":
    # Numerical array for u0 and eta0
    u0_k = np.sin(am * np.pi * x_k / Lx)  # Velocity initial condition (numerical)
    eta0_k = np.sin(am * np.pi * x_k / Lx)  # Elevation initial condition (numerical)

    # UFL expressions for u0 and eta0
    u0 = sin(am * pi * x[0] / Lx)  # Velocity initial condition (UFL expression)
    eta0 = sin(am * pi * x[0] / Lx)  # Elevation initial condition (UFL expression)

    # Initial boundary conditions for velocity and eta at x0
    u_x0 = np.sin(am * np.pi * x0 / Lx - (am * np.pi * a / Lx) * (t0 + dt))  # Inflow velocity
    eta_x0 = np.sin(am * np.pi * x0 / Lx - (am * np.pi * H0 / Lx) * (t0 + dt))  # Inflow elevation

    # Inflow boundary conditions (Constant values)
    u_in = Constant(u_x0)  # Boundary condition for velocity
    eta_in = Constant(eta_x0)  # Boundary condition for elevation

u_.interpolate(u0)
eta_.interpolate(eta0)

# === plot initial conditions ===
fig, ax = plt.subplots(2, figsize=(16, 8), layout='constrained')
eta12 = np.array([eta_.at(x) for x in x_k])  # eta12 = np.array([eta0.at(x) for x in xvals])
ax[0].plot(x_k, eta12, label=r'$\eta_0(x)$')
phi12 = np.array([u_.at(x) for x in x_k])
ax[1].plot(x_k, phi12, label=r'$u_0(x)$')

# === FLUXES ===
# Boundary fluxes

if IC == "a":
    F_1in = H0 * u_
    F_1out = H0 * u_

    F_2in = g * eta_
    F_2out = g * eta_
elif IC == "b":
    F_1in = H0 * u_in
    F_1out = H0 * u_

    F_2in = g * eta_in
    F_2out = g * eta_

F1_int = 0.5*(H0 * (u_('-') + u_('+')) - c0 * ((eta_('-') - eta_('+')))) / 2.0
F2_int = 0.5*(H0 * (u_('+') - u_('-')) + c0 * ((eta_('-') + eta_('+')))) * g / (2.0 * c0)

au = w * u_trial * dx
Lu = (w * u_ * dx +
      dtc * (w * F_2in * ds(1)  # inflow boundary integral
             - w * F_2out * ds(2)  # outflow boundary integral
             - (w('+') - w('-')) * F2_int * dS))  # Godunov flux for interior
aeta = eta__test * eta__trial * dx
Leta = (eta__test * eta_ * dx +
        dtc * (eta__test * F_1in * ds(1)  # inflow boundary integral
               - eta__test * F_1out * ds(2)  # outflow boundary integral
               - (eta__test('+') - eta__test('-')) * F1_int * dS))

# Solvers
problem1 = LinearVariationalProblem(aeta, Leta, eta)
solver1 = LinearVariationalSolver(problem1)

problem2 = LinearVariationalProblem(au, Lu, u)
solver2 = LinearVariationalSolver(problem2)

# Exact solutions for velocity and height

def u_exact(x, t, x_mid, ul, ur, etal, etar, H0, c0):
    # Left of the shock: constant velocity ul
    if x < x_mid - c0 * t:
        return ul
    # Between shock and rarefaction wave: linear interpolation
    elif x < x_mid + c0 * t and x > x_mid - c0 * t:
        return (H0 * (ul + ur) + c0 * (etal - etar)) / (2.0 * H0)
    # Right of the shock: constant velocity ur
    elif x >= x_mid + c0 * t:
        return ur

def eta_exact(x, t, x_mid, ul, ur, etal, etar, H0, c0):
    # Left of the shock: constant height etal
    if x < x_mid - c0 * t:
        return etal
    # Between shock and rarefaction wave: linear interpolation
    elif x < x_mid + c0 * t and x > x_mid - c0 * t:
        return (H0 * (ul - ur) / c0 + (etal + etar)) / 2.0
    # Right of the shock: constant height etar
    elif x >= x_mid + c0 * t:
        return etar


# Time marching loop
t = t0
step = 0
while t < 10 * Tend - 0.5 * dt:
    if IC == "b":
        u_in.assign(sin(am * pi * x0 / Lx - (am * pi * g / Lx) * (t + dt)))
        eta_in.assign(sin(am * pi * x0 / Lx - (am * pi * H0 / Lx) * (t + dt)))
    
    solver1.solve()
    solver2.solve()

    u_.assign(u)
    eta_.assign(eta)

    step += 1
    t += dt

    # Visualization for specific times
    # for 9T_p <= t < 10T_p change to 0.9
    t_output_1 = 0.3  # for 3T_p <= t < 4T_p
    t_output_2 = 0.4  
    output_step_1 = int(t_output_1 / dt)
    output_step_2 = int(t_output_2 / dt)
    if IC == "a":
        if step == output_step_1 or step == output_step_2:
            eta12 = np.array([eta_.at(x) for x in x_k])
            phi12 = np.array([u_.at(x) for x in x_k])
            eta_real_sol = [eta_exact(x, t, x_mid, u0_l, u0_r, eta0_l, eta0_r, H0, c0) for x in x_k]
            u_real_sol = [u_exact(x, t, x_mid, u0_l, u0_r, eta0_l, eta0_r, H0, c0) for x in x_k]
            #eta_diff = eta_real_sol - eta12
            #u_diff = u_real_sol - phi12
            ax[0].plot(x_k, eta_real_sol, label=fr'$\eta_{{exact}}(x,Tp={10*t:.0f})$')
            ax[0].scatter(x_k, eta12, marker='x', label=fr'$\eta_{{numerical}}(x,Tp={10*t:.0f})$')
            #ax[0].plot(x_k, eta_diff, linestyle='--', label=fr'$\eta_{{diff}}(x,Tp={10*t:.0f})$')  # Difference plot

            ax[1].plot(x_k, u_real_sol, label=fr'$u_{{exact}}(x,Tp={10*t:.0f})$')
            ax[1].scatter(x_k, phi12, marker='x', label=fr'$u_{{numerical}}(x,Tp={10*t:.0f})$')
            #ax[1].plot(x_k, u_diff, linestyle='--', label=fr'$u_{{diff}}(x,Tp={10*t:.0f})$')

    if IC == "b":
        if step == output_step_1 or step == output_step_2:
            # Compute numerical solutions at the spatial grid points
            eta12 = np.array([eta_.at(x) for x in x_k])
            phi12 = np.array([u_.at(x) for x in x_k])

            # Exact solutions for standing wave
            eta_real_sol = np.sin(am * np.pi * x_k / Lx-(am*np.pi*a/Lx)*t)  # Exact elevation (standing wave)
            u_real_sol = np.sin(am * np.pi * x_k / Lx-(am*np.pi*a/Lx)*t)   # Exact velocity (standing wave)
            eta_diff = eta_real_sol - eta12
            u_diff = u_real_sol - phi12
            # Plotting elevation (eta)
            ax[0].plot(x_k, eta_real_sol, label=fr'$\eta_{{exact}}(x,Tp={10*t:.0f})$')
            ax[0].scatter(x_k, eta12, marker='_', label=fr'$\eta_{{numerical}}(x,Tp={10*t:.0f})$')
            ax[0].plot(x_k, eta_diff, linestyle='--', label=fr'$\eta_{{diff}}(x,Tp={10*t:.0f})$')  # Difference plot

            # Plotting velocity (u)
            ax[1].plot(x_k, u_real_sol, label=fr'$u_{{exact}}(x,Tp={10*t:.0f})$')
            ax[1].scatter(x_k, phi12, marker='_', label=fr'$u_{{numerical}}(x,Tp={10*t:.0f})$')
            ax[1].plot(x_k, u_diff, linestyle='--', label=fr'$u_{{diff}}(x,Tp={10*t:.0f})$')


ax[1].set_ylabel(r'$u(x,t)$')
ax[0].set_ylabel(r'$\eta(x,t)$')
ax[1].set_xlabel(r'$x$')
ax[0].grid(True)
ax[1].grid(True)
ax[0].legend(loc='lower left')
ax[1].legend(loc='lower left')

plt.show()
plt.savefig('no4a.png', dpi = 300, bbox_inches = 'tight')
