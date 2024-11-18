from firedrake import * 
from firedrake import SpatialCoordinate
from firedrake import as_vector
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspce
import matplotlib.mlab as mlab

def Hxxtopo(H0, W, L1, Hb, slope, xk12):
    return H0 + 0.0 * xk12
def Hxxtopos(H0, W, L1, Hb, slope, xk12):
    return H0 + 0.0 *xk12

mint = 4 # Mode number
H0 = 1 # Initial water depth
grav = 1 # Dimensionless gravity
Lx = 2 * np.pi # Length of computational domain
Ld = 2 * np.pi
Ls = 1
g = 9.8
c0 = np.sqrt(g * H0)
H0s = 40
kw = np.pi * mint / Ld # Wave number
omeg = grav * kw # Angular frequency
ck = 0.2 # Wave amplitude
Nper = 4 # Number of wave periods
Tend = 10 * Nper*2.0*np.pi/omeg # Total simulation time
dt = Tend / 1000 # Time step
Nx = 100 # Number of spatial intervals
nmeas = 11 # Number of measurements
tmeas = [0.0, Tend / 10, 2 * Tend / 10 , 3 * Tend / 10 , 4 * Tend / 10 , 5 * Tend / 10 , 6 * Tend / 10 , 7 * Tend / 10 , 8 * Tend / 10 , 9 * Tend / 10 , Tend ]
tmease = 0.0
dtmeas = Tend / 32

mesh1d = IntervalMesh(Nx,Lx)
mesh = mesh1d
x, = SpatialCoordinate(mesh)

# Define the function spaces
nDG = 0
DG0 = FunctionSpace(mesh, "DG", nDG) # Define a space in which to define the functions
eta0 = Function(DG0 , name = "eta0") # Inititalise the function for intitial eta
uve0 = Function(DG0, name = "uve0") # Initialise the function for initial u
eta1 = Function(DG0, name = "eta1")
uve1 = Function(DG0, name = "uve1")

eta_trial = TrialFunction(DG0)
uve_trial = TrialFunction(DG0)
deta_test = TestFunction(DG0)
duve_test = TestFunction(DG0)

time = 0.0

eta0 = Function(DG0).interpolate( ck*cos((mint*np.pi*x)/Ld)*cos(omeg*time) )
uve0 = Function(DG0).interpolate( (omeg*ck/H0)*(Ld/(mint*np.pi))*sin(mint*np.pi*x/Ld)*sin(omeg*time) )
Hx = Hxxtopo(H0, 0.0, 0.0, 0.0, 0.0, x)

t = time 
t_ = Constant(t)
smallfac = 10 ** (-10)

nx = Nx
xsmall = 0.0 * 10 ** (-6)
xvals = np.linspace(0 + xsmall, Lx - xsmall, nx)

fig, (ax1, ax2) = plt.subplots(2)
tsize = 14
ax1.set_ylabel(r'$\eta(x,t)$ ',fontsize=tsize)
ax1.grid()
ax2.set_xlabel(r'$x$ ',fontsize=tsize)
ax2.set_ylabel(r'$u(x,t)$ ',fontsize=tsize)
ax2.grid()

Hxvals = Hxxtopos(H0, 0.0 , 0.0, 0.0, 0.0, xvals)

eta12 = np.array([eta0.at(x) for x in xvals])
phi12 = np.array([uve0.at(x) for x in xvals])

Nkel = 40 # Number of elements
Xce = np.zeros(Nkel+1) # Cell edges

dxx = Lx / Nkel
Hxt = H0 + 0.0 * Xce

for jj in range(0,Nkel+1):
    Xce[jj] = jj * dxx
    Hxt[jj] = Hxxtopos(-H0, 0.0, 0.0, 0.0, 0.0, Xce[jj])


n = as_vector([1.0])
    
a_masseta = deta_test * eta_trial * dx 
u20 = Constant(0.0)
u10 = Constant(0.0)

eta0vals = np.array([eta0.at(x) for x  in xvals])
uve0vals = np.array([uve0.at(x) for x in xvals])

Fetaflux = 0.5 * n[0] * ((c0 / H0) * (uve0("-") - uve0("+")) + (eta0("-") + eta0("+")))
Fetafluxbcl =  1e-16
Fetafluxbcr =   1e-16
etarhs = deta_test * eta0 * dx - dt*Fetaflux*(deta_test('+')-deta_test('-'))*dS # derivative of test function zero for DG0
etarhs = etarhs + dt*Fetafluxbcl*deta_test*ds(1) - dt*Fetafluxbcr*deta_test*ds(2) # note ugly plus sign in ds(1) intuition not in the maths; due to odd normals in 1d
eta_problem = LinearVariationalProblem(a_masseta, etarhs, eta1)

a_massuve = duve_test * uve_trial * dx 

Fuveflux = 0.5 * n[0] * ((uve0("-") + uve0("+")) + (c0 / H0) * (eta0("-") - eta0("+")))
Fuvefluxbcl = - (uve0vals[0] ) +1e-16
Fuvefluxbcr = - (uve0vals[-1] ) +1e-16
uverhs = duve_test * uve0 * dx  - dt*Fuveflux*(n[0]*(duve_test('+')-duve_test('-')))*dS
uverhs = uverhs + dt*Fuvefluxbcl*(n[0]*duve_test)*ds(1) - dt*Fuvefluxbcr*(n[0]*duve_test)*ds(2)  # note ugly plus sign in ds(1) not in the maths; due to odd normals in 1d
uve_problem = LinearVariationalProblem(a_massuve, uverhs, uve1)

params= {"ksp_type": "preonly", "pc_type": "jacobi"}
solv1 = LinearVariationalSolver(eta_problem, solver_parameters=params)
solv2 = LinearVariationalSolver(uve_problem, solver_parameters=params)

t = 0

while t <= Tend:

    t = t
    solv1.solve()
    solv2.solve()
    eta0.assign(eta1)
    uve0.assign(uve1) 
    eta0vals = np.array([eta0.at(x) for x  in xvals])
    uve0vals = np.array([uve0.at(x) for x in xvals])
    Fetaflux = 0.5 * n[0] * ((c0 / H0) * (uve0("-") - uve0("+")) + (eta0("-") + eta0("+")))
    Fetafluxbcl = 1e-16
    Fetafluxbcr = 1e-16
    etarhs = deta_test * eta0 * dx - dt*Fetaflux*(deta_test('+')-deta_test('-'))*dS # derivative of test function zero for DG0
    etarhs = etarhs + dt*Fetafluxbcl*deta_test*ds(1) - dt*Fetafluxbcr*deta_test*ds(2) # note ugly plus sign in ds(1) intuition not in the maths; due to odd normals in 1d
    eta_problem = LinearVariationalProblem(a_masseta, etarhs, eta1)

    Fuveflux = 0.5 * n[0] * ((uve0("-") + uve0("+")) + (c0 / H0) * (eta0("-") - eta0("+")))
    Fuvefluxbcl =  -uve0vals[0] 
    Fuvefluxbcr = -uve0vals[-1] 
    uverhs = duve_test * uve0 * dx  - dt*Fuveflux*(n[0]*(duve_test('+')-duve_test('-')))*dS
    uverhs = uverhs + dt*Fuvefluxbcl*(n[0]*duve_test)*ds(1) - dt*Fuvefluxbcr*(n[0]*duve_test)*ds(2)  # note ugly plus sign in ds(1) not in the maths; due to odd normals in 1d
    uve_problem = LinearVariationalProblem(a_massuve, uverhs, uve1)
    
    t = t + dt

a = len(xvals)
topo = np.zeros(a)
for k in range(a):
    topo[k] = -H0

eta12 = np.array([eta0.at(x) for x in xvals])
phi12 = np.array([uve0.at(x) for x in xvals])
ax1.plot(xvals, eta12 , label = f'eta after {dt} seconds')
ax1.plot(xvals,topo, 'k', lw = 3)
ax2.plot(xvals, phi12)

plt.show()