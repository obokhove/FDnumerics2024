from firedrake import *
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Define mesh resolutions and polynomial orders
# These are the parameters for the mesh sizes (resolutions) and the polynomial degrees to be tested.
mesh_resolutions = [16, 32, 64, 128]
polynomial_orders = [1, 2, 3, 4]

# Store errors for plotting
# Errors for both methods will be stored in dictionaries indexed by mesh resolution.
errors_method1 = {nx: [] for nx in mesh_resolutions}
errors_method2 = {nx: [] for nx in mesh_resolutions}

# Loop over mesh resolutions and polynomial degrees
# Iterate through each combination of mesh resolution and polynomial degree.
for nx in mesh_resolutions:
    for p in polynomial_orders:
        print(f"Solving for mesh resolution {nx}x{nx} and polynomial degree {p}")
        
        # Generate mesh and function space
        # A unit square mesh is generated for the specified resolution, and a continuous Galerkin (CG) function space is created.
        mesh = UnitSquareMesh(nx, nx, quadrilateral=True)
        V = FunctionSpace(mesh, 'CG', p)
        
        # Define test and trial functions
        # These functions represent the trial (u) and test (v) functions in the weak form of the equation.
        u = TrialFunction(V)
        v = TestFunction(V)
        x, y = SpatialCoordinate(mesh)
        
        # Define source term and exact solution
        # f is the source term, and u_exact is the known exact solution for comparison.
        f = Function(V).interpolate(2*pi**2*sin(pi*x)*cos(pi*y))
        u_exact = Function(V).interpolate(sin(pi*x)*cos(pi*y))
        
        # Weak form for Method 1
        # The weak form is defined for Method 1, involving gradient terms for the trial and test functions.
        a = inner(grad(u), grad(v)) * dx
        L = f * v * dx
        u_1 = Function(V, name='u_1')
        
        # Boundary conditions
        # Dirichlet boundary conditions are set to zero at both x=0 and x=1 boundaries.
        bc_x0 = DirichletBC(V, Constant(0), 1)
        bc_x1 = DirichletBC(V, Constant(0), 2)
        
        # Solve Method 1
        # Solve the linear system for Method 1 using the defined weak form and boundary conditions.
        solve(a == L, u_1, solver_parameters={'ksp_type': 'cg', 'pc_type': 'none'}, bcs=[bc_x0, bc_x1])
        
        # Method 2: Variational derivative approach
        # A variational approach is used to minimize the energy functional and find the solution for Method 2.
        u_2 = Function(V, name='u_2')
        Ju = (0.5 * inner(grad(u_2), grad(u_2)) - u_2 * f) * dx
        F = derivative(Ju, u_2, du=v)
        solve(F == 0, u_2, bcs=[bc_x0, bc_x1])

        # Difference between numerical solution and exact solution
        # Compute the absolute differences between the solutions from both methods and the exact solution.
        u_diff_1 = Function(V, name='u_diff_1')
        u_diff_2 = Function(V, name='u_diff_2')

        # Project the absolute difference onto the function space
        # The absolute differences are projected onto the same function space for further analysis.
        u_diff_1_projector = project(abs(u_1 - u_exact), V)
        u_diff_2_projector = project(abs(u_2 - u_exact), V)
        # Assign the result to the Function objects
        u_diff_1.assign(u_diff_1_projector)
        u_diff_2.assign(u_diff_2_projector)

        # Save solutions to a .pvd file
        # The solutions are saved to VTK files for visualization and further analysis.
        outfile = VTKFile(f'output_h{1/nx}_p{p}.pvd')
        outfile.write(u_1, u_2, u_exact, u_diff_1, u_diff_2)
        # Compute L2 errors
        # L2 norm errors are computed for both methods to assess the accuracy of the numerical solutions.
        L2_1 = sqrt(assemble(dot(u_1 - u_exact, u_1 - u_exact) * dx))
        L2_2 = sqrt(assemble(dot(u_2 - u_exact, u_2 - u_exact) * dx))
        
        # Append errors for plotting
        # The L2 errors are stored for each mesh resolution and polynomial degree.
        errors_method1[nx].append(L2_1)
        errors_method2[nx].append(L2_2)
        
        # Print results
        # The L2 errors for both methods are printed to the console for each combination of mesh resolution and polynomial degree.
        print(f"Mesh resolution: Δx = {1/nx}, Polynomial degree: {p}")
        print(f"L2 error: Method 1 = {L2_1:.5e}, Method 2 = {L2_2:.5e}")

# Plotting L2 errors for each mesh resolution
# A 2x2 grid of subplots is created to visualize the L2 errors for both methods across different mesh resolutions and polynomial degrees.
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
'''
for i, nx in enumerate(mesh_resolutions):
    ax = axes[i]
    
    # Plot Method 1 and Method 2 errors
    ax.plot(polynomial_orders, errors_method1[nx], 'o-', label="Method 1")
    ax.plot(polynomial_orders, errors_method2[nx], 's-', label="Method 2")
    
    # Add error values as text annotations
    for j, p in enumerate(polynomial_orders):
        ax.text(p, errors_method1[nx][j], f"{errors_method1[nx][j]:.2e}", 
                ha='center', va='bottom', fontsize=8, color='blue')
        ax.text(p, errors_method2[nx][j], f"{errors_method2[nx][j]:.2e}", 
                ha='center', va='top', fontsize=8, color='orange')
    
    # Plot settings
    ax.set_title(f"Mesh Resolution: {nx}x{nx}")
    ax.set_xlabel("Polynomial Degree (p)")
    ax.set_ylabel("L2 Error")
    ax.set_ylim(-0.0001, 0.0017)
    ax.legend()
    ax.grid()
'''
for i, p in enumerate(polynomial_orders):
    ax = axes[i]
    
    # Plot Method 1 and Method 2 errors for each mesh resolution (nx)
    ax.plot(mesh_resolutions, [errors_method1[nx][i] for nx in mesh_resolutions], 'o-', label="Method 1")
    ax.plot(mesh_resolutions, [errors_method2[nx][i] for nx in mesh_resolutions], 's-', label="Method 2")
    
    # Add error values as text annotations for each point
    # The error values are annotated on the plot for better visualization.
    for j, nx in enumerate(mesh_resolutions):
        ax.text(nx, errors_method1[nx][i], f"{errors_method1[nx][i]:.2e}", 
                ha='center', va='bottom', fontsize=8, color='blue')
        ax.text(nx, errors_method2[nx][i], f"{errors_method2[nx][i]:.2e}", 
                ha='center', va='top', fontsize=8, color='orange')
    
    # Plot settings
    # Set plot title, labels, and formatting for better visualization.
    ax.set_title(f"Polynomial Order: {p}")
    ax.set_xlabel("Mesh Resolution (nx)")
    ax.set_ylabel("L2 Error")
    ax.set_xticks([16, 32, 64, 128])
    ax.set_xticklabels([str(nx) for nx in [16, 32, 64, 128]]) 
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.legend()
    ax.grid()

plt.tight_layout()
# Save the plot as a PNG image
plt.savefig("error_plots_with_values.png")
plt.show()