#
# Solution by Firedrake FEM-CG of a Poisson equatiom
#
from firedrake import *

nx = ny = 128 # Try various mesh resolutions, starting coarse, say 16x16 etc.

mesh = UnitSquareMesh(nx,ny,quadrilateral=True)
# Quadrilateral regular mesh made: https://www.firedrakeproject.org/firedrake.html#firedrake.utility_meshes.UnitSquareMesh
# Alternatively use gmsh: 

V = FunctionSpace(mesh, 'CG', 1) # Piecewise linear continuous Galerkin function space or polynomials
# See: https://www.firedrakeproject.org/variational-problems.html

#
# Method 1: construct the weak form manually by multiplying and manipulating the Poisson equation and solve the linear system
#
u = TrialFunction(V) # The unknown or variable u(x,y)
v = TestFunction(V)  # The testfunction of u, which may be better called delu or deltau

x, y = SpatialCoordinate(mesh) # Mesh coordinates

f = Function(V).interpolate(2*pi**2*sin(pi*x)*cos(pi*y)) # The given function f(x,y)

a = (inner(grad(u),grad(v)))*dx # Step 2/3: The weak form first term
L = (f*v)*dx # Step 2/3: The weak form second term; dx is the infinitesimal piece in the damain here: dx*dy=dA with area A.

u_1 = Function(V, name='u_1') # Name of solution for first method

bc_x0 = DirichletBC(V, Constant(0), 1) # Dirichlet boundary conditions imposed 
bc_x1 = DirichletBC(V, Constant(0), 2) # Dirichlet boundary conditions imposed 
# See: https://www.firedrakeproject.org/firedrake.html#firedrake.bcs.DirichletBC
# The homogeneous Neumann boundary conditions are "automatically" included, i.e. do not need anything explicit

solve(a == L, u_1, solver_parameters={'ksp_type': 'cg', 'pc_type': 'none'}, bcs=[bc_x0,bc_x1]) # Step 4: the solution assigned to u1

#
# Method 2: generate the weak form via "derivative()" of the Ritz-Galerkin integral or variational principle and solve the nonlinear system
#
u_2 = Function(V, name='u_2') # Name of solution for first method

Ju = (0.5*inner(grad(u_2),grad(u_2)) - u_2*f)*dx # f->ULF? Step 2

F = derivative(Ju, u_2, du=v) # Step 2/3: The weak form generated

solve(F == 0, u_2, bcs=[bc_x0, bc_x1]) # Step 4: the solution assigned to u2

# 
# Post-processing: Use Paraview to visualise
# 
outfile = File('output.pvd')
outfile.write(u_1, u_2)

f.interpolate(sin(pi*x)*cos(pi*y))
L2_1 = sqrt(assemble(dot(u_1 - f, u_1 - f) * dx)) # L2 error solution u1
L2_2 = sqrt(assemble(dot(u_2 - f, u_2 - f) * dx)) # L2 error solution u2
L2 = sqrt(assemble(dot(u_2 - u_1, u_2 - u_1) * dx)) # L2 error difference
print(f'Mesh resolution: Δx = {1/nx}')
print(f'L2 error: Method1 = {L2_1}, Method2 = {L2_2}')
print(f'L2 norm between the two results: {L2}')
