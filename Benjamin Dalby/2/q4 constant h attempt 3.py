#Onno - this code is adapted from your sweDGFV.py. Given your footnote I hope this ok.
#Uses Gudonov dt scheme for standing wave solution
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["OMP_NUM_THREADS"] = "1"
from firedrake import *

t = 0
H_0 = 1
mint = 4
grav = 9.81
H0 = 1.0
Lx = 2.0*np.pi          #Length in x
Ld = Lx
kw = np.pi*mint/Ld
omeg = grav*kw
ck = 0.2
Nper = 4
Tend = Nper*2.0*np.pi/omeg
print('Tend',Tend)
dt = Tend/5000
dtmeas = Tend/32
Nx = 500            #Number of cells
t_ = Constant(t)
xvals = np.linspace(0, Lx, Nx)

mesh = IntervalMesh(Nx, Lx)
x, = SpatialCoordinate(mesh)

#Define functions
nDG = 0
DG0 = FunctionSpace(mesh, "DG", nDG)
eta0 = Function(DG0, name="eta0")       #eta at current time
uve0 = Function(DG0, name="uve0")       #u at current time
eta1 = Function(DG0, name="eta1")       #eta at n+1
uve1 = Function(DG0, name="uve1")       #u at n+1
eta_trial = TrialFunction(DG0)
uve_trial = TrialFunction(DG0)
deta_test = TestFunction(DG0)
duve_test = TestFunction(DG0)

H_0c = Constant(H_0)
n = as_vector([1.0]) 
a_masseta = deta_test * eta_trial * dx
dtc = Constant(dt)
C_0 = grav * H_0c

Hx = Constant(H_0)                #Adapting to make Hx constant everywhere

#Initial value
tijd = 0
eta0 = Function(DG0).interpolate( ck*cos(mint*pi*x/Ld)*cos(omeg*tijd) ) # u1 is eta
uve0 = Function(DG0).interpolate( (omeg*ck/H0)*(Ld/(mint*pi))*sin(mint*pi*x/Ld)*sin(omeg*tijd) ) # u2 is velocity

Fetaflux = 0.5*((Hx/C_0)*(uve0('-') - uve0('+')) + (eta0('-') - eta0('+')))
Fetafluxbcl = eta0
Fetafluxbcr = eta0

etarhs = deta_test * eta0 * dx
etarhs = etarhs + dt*(Fetafluxbcl*deta_test*ds(1) - Fetafluxbcr*deta_test*ds(2) - ((deta_test('+') + deta_test('-'))*Fetaflux*dS))
eta_problem = LinearVariationalProblem(a_masseta, etarhs, eta1)

a_massuve = duve_test * uve_trial * dx # inner product 
Fuveflux = 0.5*(Hx*(uve0('+') + uve0('-')) + C_0(eta0('-') + eta0('+')))
Fuvefluxbcl =  Hx*(-1*uve0)
Fuvefluxbcr =  Hx*(-1*uve0)

uverhs = duve_test * uve0 * dx 
uverhs = etarhs + dt*(Fuvefluxbcl*duve_test*ds(1) + Fuvefluxbcr*duve_test*ds(2) - ((duve_test('+') - duve_test('-'))*Fuveflux*dS))
uve_problem = LinearVariationalProblem(a_massuve, uverhs, uve1)

# Since we have arranged that the matrix A is diagonal, we can invert it with a single application of Jacobi iteration. We select this here using appropriate solver parameters, which tell PETSc to construct a solver which just applies a single step of Jacobi preconditioning. See: https://www.firedrakeproject.org/demos/higher_order_mass_lumping.py.html
#params= {"ksp_type": "preonly", "pc_type": "jacobi"}
solv1 = LinearVariationalSolver(eta_problem)
solv2 = LinearVariationalSolver(uve_problem)

#t_.assign(t)

while t <= Tend:
    t += dt
    #t_.assign(t)

    solv1.solve()
    solv2.solve()
    eta0.assign(eta1)
    uve0.assign(uve1)

eta = np.array([eta0.at(x) for x in xvals])
u = np.array([uve0.at(x) for x in xvals])

eta_exact = ck*np.cos(mint*np.pi*xvals/Ld)*np.cos(omeg*t)
u_exact = (omeg*ck/H0)*(Ld/(mint*np.pi))*np.sin(mint*np.pi*xvals/Ld)*np.sin(omeg*t) 

#eta_exact = eta_exact[::5]     #take every fifth value, to make visualisation nicer
#u_exact = u_exact[::5]
#eta = eta[::5]
#u = u[::5]
#xvals = xvals[::5]

plt.plot(xvals, eta, label="$\eta$")
plt.plot(xvals, u, label="u")
#plt.scatter(xvals,eta_exact,label='exact eta',marker='o')
#plt.scatter(xvals, u_exact, label="exact u",marker='x')
plt.xlabel("x value")
plt.title(f"Standing wave solution for time period {Nper}T")
plt.legend()
plt.show()
plt.savefig('/mnt/c/Users/dmcx3376/Documents/Foundations/Numerical methods/2/plot1.png')