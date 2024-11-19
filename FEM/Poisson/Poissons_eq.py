from firedrake import *

nx = ny = 8

mesh = UnitSquareMesh(nx,ny,quadrilateral=True)
#https://www.firedrakeproject.org/firedrake.html#firedrake.utility_meshes.UnitSquareMesh

V = FunctionSpace(mesh, 'CG', 1)

u = TrialFunction(V)
v = TestFunction(V)

x, y = SpatialCoordinate(mesh)

f = Function(V).interpolate(2*pi**2*sin(pi*x)*cos(pi*y))

a = (inner(grad(u),grad(v)))*dx
L = (f*v)*dx

u_h = Function(V, name='u')

bc_x0 = DirichletBC(V, Constant(0), 1)
bc_x1 = DirichletBC(V, Constant(0), 2)
#https://www.firedrakeproject.org/firedrake.html#firedrake.bcs.DirichletBC

solve(a == L, u_h, solver_parameters={'ksp_type': 'cg', 'pc_type': 'none'}, bcs=[bc_x0,bc_x1])

outfile = File('output.pvd')
outfile.write(u_h)

f.interpolate(sin(pi*x)*cos(pi*y))
L2 = sqrt(assemble(dot(u_h - f, u_h - f) * dx))
print(f'Mesh resolution: Δx = {1/nx}, L2 error = {L2}')
