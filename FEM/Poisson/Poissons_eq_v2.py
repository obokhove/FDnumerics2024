from firedrake import *

nx = ny = 128

mesh = UnitSquareMesh(nx,ny,quadrilateral=True)
#https://www.firedrakeproject.org/firedrake.html#firedrake.utility_meshes.UnitSquareMesh

V = FunctionSpace(mesh, 'CG', 1)

# Method 1: construct the weak form manually and solve the linear system
u = TrialFunction(V)
v = TestFunction(V)

x, y = SpatialCoordinate(mesh)

f = Function(V).interpolate(2*pi**2*sin(pi*x)*cos(pi*y))

a = (inner(grad(u),grad(v)))*dx
L = (f*v)*dx

u_1 = Function(V, name='u_1')

bc_x0 = DirichletBC(V, Constant(0), 1)
bc_x1 = DirichletBC(V, Constant(0), 2)
#https://www.firedrakeproject.org/firedrake.html#firedrake.bcs.DirichletBC

solve(a == L, u_1, solver_parameters={'ksp_type': 'cg', 'pc_type': 'none'}, bcs=[bc_x0,bc_x1])

# Method 2: generate the weak form via "derivative()" and solve the nonlinear system
u_2 = Function(V, name='u_2')

Ju = (0.5*inner(grad(u_2),grad(u_2)) - u_2*f)*dx #f->ULF?

F = derivative(Ju, u_2, du=v)

solve(F == 0, u_2, bcs=[bc_x0, bc_x1])

# post-processing
outfile = File('output.pvd')
outfile.write(u_1, u_2)

f.interpolate(sin(pi*x)*cos(pi*y))
L2_1 = sqrt(assemble(dot(u_1 - f, u_1 - f) * dx))
L2_2 = sqrt(assemble(dot(u_2 - f, u_2 - f) * dx))
L2 = sqrt(assemble(dot(u_2 - u_1, u_2 - u_1) * dx))
print(f'Mesh resolution: Δx = {1/nx}')
print(f'L2 error: Method1 = {L2_1}, Method2 = {L2_2}')
print(f'L2 norm between the two results: {L2}')
