#
# Solution by Firedrake FEM-CG of a Poisson equatiom
#
#%%
from firedrake import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import ScalarFormatter


# cols = ['p', 'nx', 'L2_1', 'L2_2', 'L2']
L2_err_data = []
nx = ny = 16 # Try various mesh resolutions, starting coarse, say 16x16 etc. 16,32,64,128
p_value = 4


for p_value in [1,2,3,4]:
    for nx in [16,32,64,128]:
        ny = nx
        mesh = UnitSquareMesh(nx,ny,quadrilateral=True)
        # Quadrilateral regular mesh made: https://www.firedrakeproject.org/firedrake.html#firedrake.utility_meshes.UnitSquareMesh
        # # Alternatively use gmsh: 
        
        V = FunctionSpace(mesh, 'CG', p_value) # Piecewise linear continuous Galerkin function space or polynomials
        # See: https://www.firedrakeproject.org/variational-problems.html
        
        #
        # # Method 1: construct the weak form manually by multiplying and manipulating the Poisson equation and solve the linear system
        
        
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
        # # The homogeneous Neumann boundary conditions are "automatically" included, i.e. do not need anything explicit
        
        solve(a == L, u_1, solver_parameters={'ksp_type': 'cg', 'pc_type': 'none'}, bcs=[bc_x0,bc_x1]) # Step 4: the solution assigned to u1
        # print(u_1)
        # #
        # # Method 2: generate the weak form via "derivative()" of the Ritz-Galerkin integral or variational principle and solve the nonlinear system
        
        #
        u_2 = Function(V, name='u_2') # Name of solution for first method
        
        Ju = (0.5*inner(grad(u_2),grad(u_2)) - u_2*f)*dx # f->ULF? Step 2
        
        F = derivative(Ju, u_2, du=v) # Step 2/3: The weak form generated
        
        solve(F == 0, u_2, bcs=[bc_x0, bc_x1]) # Step 4: the solution assigned to u2
        
        # 
        # # Post-processing: Use Paraview to visualise
        # # See https://www.firedrakeproject.org/visualisation.html#creating-output-files
        outfile = VTKFile('output.pvd')
        outfile.write(u_1, u_2)
        
        f.interpolate(sin(pi*x)*cos(pi*y))
        L2_1 = sqrt(assemble(dot(u_1 - f, u_1 - f) * dx)) # L2 error solution u1
        L2_2 = sqrt(assemble(dot(u_2 - f, u_2 - f) * dx)) # L2 error solution u2
        L2 = sqrt(assemble(dot(u_2 - u_1, u_2 - u_1) * dx)) # L2 error difference
        
        L2_err_data.append ([p_value, nx, L2_1,L2_2,L2])
        # print(f'Mesh resolution: Δx = {1/nx}')
        # print(f'L2 error: Method1 = {L2_1}, Method2 = {L2_2}')
        # print(f'L2 norm between the two results: {L2}')
        
        
        
        ####### exact value ############
        def u_ext_func (x,y):
            return np.sin(np.pi*x)*np.cos(np.pi*y)
        #### Saving u1 and u2 i a form that is acceptable for  plt.contourf
        x_values = np.linspace(0, 1, nx)
        y_values = np.linspace(0, 1, ny)
        u_1_vec = []
        u_2_vec = []
        u_e_vec = []
        u_1_e_vec = []
        u_2_e_vec = []
        u_1_2_vec = []
        for y_fix in y_values:
            u_e_row = [u_ext_func(x,y_fix) for x in x_values]
            u_1_row = [u_1.at((x, y_fix)).item() for x in x_values]
            u_2_row = [u_2.at((x, y_fix)).item() for x in x_values]
            u_1_e_vec.append( [abs(u_e_row[i] - u_1_row[i]) for i in range (len(u_e_row))])
            u_2_e_vec.append( [abs(u_e_row[i] - u_2_row[i]) for i in range (len(u_e_row))])
            u_1_2_vec.append( [abs(u_2_row[i] - u_1_row[i]) for i in range (len(u_e_row))])
            u_e_vec.append(u_e_row)
            u_1_vec.append(u_1_row)
            u_2_vec.append(u_2_row)
                
        # print(u_1_row)
            
                
        ##### Plots ##########
        
        # Define custom colormap from blue to red
        custom_cmap = LinearSegmentedColormap.from_list("blue_red", ["blue", "white", "red"])
        
        
        # Define square domain: x and y from 0 to 1
        X, Y = np.meshgrid(x_values, y_values)
        
        
        
        h_plot = 3
        v_plot = 2
        plot_widt = 12
        contours_vec = []
        fig, ax = plt.subplots(v_plot, h_plot,  sharey=True, figsize=(plot_widt, 6)) #figsize=(plot_widt, plot_widt*v_plot/h_plot)
        # print( plot_widt*v_plot/h_plot)
        contours_vec.append(ax[0,0].contourf(X, Y, u_1_vec, levels=20, cmap=custom_cmap))
        ax[0,0].set_title(r'a) $u_{h,1}$, Method 1',loc='left')
        ax[0,0].set_ylabel('y')
        ax[0,0].tick_params(axis='x',labelbottom=False)
        fig.colorbar(contours_vec[0],label=r'$u_{h,1}$', ax = ax[0,0])
        
        contours_vec.append(ax[0,1].contourf(X, Y, u_2_vec, levels=20, cmap=custom_cmap))
        ax[0,1].set_title(r'b) $u_{h,2}$, Method 2',loc='left')
        ax[0,1].tick_params(axis='x',labelbottom=False)
        fig.colorbar(contours_vec[1],label=r'$u_{h,2}$', ax = ax[0,1])
        
        contours_vec.append(ax[0,2].contourf(X, Y, u_e_vec, levels=20, cmap=custom_cmap))
        ax[0,2].set_title(r'c) $u_e$',loc='left')
        ax[0,2].tick_params(axis='x',labelbottom=False)
        fig.colorbar(contours_vec[2],label=r'$u_e$', ax = ax[0,2])
        ### errors ###
        max_error = max(max(max(u_1_e_vec)),max(max(u_2_e_vec)))
        contours_vec.append(ax[1,0].contourf(X, Y, u_1_e_vec, levels=20, cmap='Greens', vmax=max_error))
        ax[1,0].set_title(r'd) $|u_{h,1} - u_e|$, Method 1',loc='left')
        ax[1,0].set_xlabel('x')
        ax[1,0].set_ylabel('y')
        cbar = fig.colorbar(contours_vec[3],label=r'$|u_{h,1} - u_e|$', ax = ax[1,0])
        cbar.formatter = ScalarFormatter()
        cbar.formatter.set_scientific(True)  # Enable scientific notation
        cbar.formatter.set_powerlimits((-2, 2))  # Control when to switch to scientific notation
        cbar.ax.yaxis.set_major_formatter(cbar.formatter)
        cbar.update_ticks()
        
        
        contours_vec.append(ax[1,1].contourf(X, Y, u_2_e_vec, levels=20, cmap='Greens', vmax=max_error))
        ax[1,1].set_title(r'e) $|u_{h,2} - u_e|$, Method 2',loc='left')
        ax[1,1].set_xlabel('x')
        cbar = fig.colorbar(contours_vec[4],label=r'$|u_{h,2} - u_e|$', ax = ax[1,1])
        cbar.formatter = ScalarFormatter()
        cbar.formatter.set_scientific(True)  # Enable scientific notation
        cbar.formatter.set_powerlimits((-2, 2))  # Control when to switch to scientific notation
        cbar.ax.yaxis.set_major_formatter(cbar.formatter)
        cbar.update_ticks()
        
        
        contours_vec.append(ax[1,2].contourf(X, Y, u_1_2_vec, levels=20, cmap='Greens', vmax=max_error))
        ax[1,2].set_title(r'f) $|u_{h,1} - u_{h,2}|$',loc='left')
        ax[1,2].set_xlabel('x')
        cbar = fig.colorbar(contours_vec[5],label=r'$|u_{h,1} - u_{h,2}|$', ax = ax[1,2])
        
        
        fig.suptitle(r'p = {}, {}X{} grid, $\Delta x = \Delta y$ = {:.2e}'.format(p_value,nx,ny,1.0/nx), fontsize = 16)
        plt.show()
        # plt.subplots_adjust( hspace = 1.0 ) #hspace=0, top = 0.94
        plt.savefig(f'images/countourn_plot_{p_value}_{nx}X_{ny}.png', dpi = 500, bbox_inches = 'tight')
        

np.savetxt("L2Error.txt", L2_err_data, delimiter=",")


