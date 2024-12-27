import numpy as np
import matplotlib.pyplot as plt


x_left  = -1.0  # left boundary
x_right =  1.0  # right boundary
J = 40          # number of subintervals
dx = (x_right - x_left) / J  # => 2.0/40 = 0.05
x = np.linspace(x_left, x_right, J+1)  # 41 points from -1 to +1

dt = 0.001   # time step (example)
Nt = 50      # total time steps (example)

def initial_condition(x):
    """
    Piecewise definition:
      u0(x) = 1 + x   if -1 <= x <= 0
              1 - x   if  0 < x <= 1
    """
    u0 = np.zeros_like(x)
    mask_left  = (x >= -1.0) & (x <= 0.0)   # -1 <= x <= 0
    mask_right = (x >  0.0) & (x <= 1.0)    #  0 < x <= 1
    # We'll define x=0 in the 'left' part or you can split it differently
    u0[mask_left]  = 1.0 + x[mask_left]
    u0[mask_right] = 1.0 - x[mask_right]
    return u0

def explicit_diffusion_scheme(Nt, dt, dx, J):
    
    
    # Initialize
    u = initial_condition(x)   # shape = (J+1,)
    
    # To store snapshots at certain times
    stored_solutions = [u.copy()]
    
    # Time stepping
    for n in range(Nt):
        u_new = u.copy()
        
        # Update interior points: i=1..J-1
        for i in range(1, J):
            # Central difference for second derivative
            u_new[i] = (u[i] 
                        + dt/dx**2 * (u[i+1] - 2*u[i] + u[i-1]))
        
        # Dirichlet BCs at x=-1 => i=0 and x=1 => i=J
        u_new[0] = 0.0
        u_new[J] = 0.0
        
        # Advance in time
        u = u_new
        
        # Store solution at some steps
        if n in [0, 10, 20, 30, 40, 50]:
            stored_solutions.append(u.copy())
    
    return stored_solutions


def main():
    solutions = explicit_diffusion_scheme(Nt, dt, dx, J)
    
    plt.figure(figsize=(10,4))
    
    # Plot each stored snapshot
    labels = ["t=0 (IC)", "n=10", "n=20", "n=30", "n=40", "n=50"]
    for k, sol in enumerate(solutions):
        plt.plot(x, sol, marker='o', label=labels[k])
    
    plt.title("Explicit on [-1,1] with dx=0.05, dt=0.001")
    plt.xlabel("x")
    plt.ylabel("$u_j^{n}$")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
