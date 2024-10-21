#Aly Ilyas
import numpy as np
import matplotlib.pyplot as plt

# Parameters
L_p = -1.0  # Left boundary
L = 1.0     # Right boundary
J = 100     # Number of spatial points
Delta_x = (L - L_p) / J  # Spatial step size
x = np.linspace(L_p, L, J + 1)  # Spatial grid points
T = 1     # Total time for simulation
a = 1     # Advection constan

# Time step parameters
Delta_t_values = [0.01, 0.02, 0.025]  # Different time step sizes
theta_values = [0, 0.5, 1]           # Different theta values
epsilon = 0.001                        # Diffusion constant

# Define Legendre polynomials
def phi_0(x):
    return np.ones_like(x)

def phi_1(x):
    return x

def phi_2(x):
    return (3/2) * x**2 - 1/2

def phi_3(x):
    return (5/2) * x**3 - (3/2) * x

# Random coefficients
np.random.seed(42)  # For reproducibility
b_k = np.random.uniform(0, 1, 4)

# Print the coefficients
print("Random coefficients b_k:", b_k)

# Compute C such that the initial condition is non-negative
def compute_C():
    values = np.vstack([phi_0(x), phi_1(x), phi_2(x), phi_3(x)]).T  # Stack the polynomial evaluations
    return max(-np.sum(b_k * values, axis=1))

C = compute_C()

# Initial condition
def initial_condition(x):
    polynomial_sum = np.sum([b_k[k] * phi_k(x) for k, phi_k in enumerate([phi_0, phi_1, phi_2, phi_3])], axis=0)
    return (1 - x)**4 * (1 + x) * (polynomial_sum + C)

# Function to implement the explicit scheme
def explicit_scheme(Delta_t, max_time, theta):
    mu = epsilon * Delta_t / Delta_x**2  # Diffusion coefficient
    gamma = a * Delta_t / Delta_x  # Advection coefficient
    u = initial_condition(x)
    u_new = np.zeros_like(u)
    time_profiles = []

    current_time = 0.0  # Start from time = 0
    time_intervals = np.arange(0, max_time + Delta_t, Delta_t)  # Time intervals at which to store results

    while current_time <= max_time:
        # Store solution at specific time intervals
        if current_time in time_intervals:
            time_profiles.append(u.copy())

        # Apply the explicit scheme for internal points with theta incorporated
        for j in range(1, J):
            u_new[j] = u[j] + theta * (mu * (u[j-1] - 2 * u[j] + u[j+1])) + (1 - theta) * (u[j] + gamma * (u[j+1] - u[j]))

        # Apply boundary conditions
        u_new[0] = 0  # u(L_p, t) = 0
        u_new[-1] = 0  # u(L, t) = 0

        # Update the solution
        u = u_new.copy()

        # Update the current time
        current_time += Delta_t

    return time_profiles

# Main simulation loop for different values of Delta_t
for Delta_t in Delta_t_values:
    for theta in theta_values:
        time_profiles = explicit_scheme(Delta_t, T, theta)

        # Calculate mu for the current Delta_t
        mp = a * Delta_t / Delta_x

        # Plotting the results
        plt.figure(figsize=(10, 6))
        for idx, profile in enumerate(time_profiles):
            plt.plot(x, profile, label=f'Time = {idx * Delta_t:.6f}')
        plt.title(f'Time Profiles for CFL = {mp:.4f}, θ = {theta}')
        plt.xlabel('x')
        plt.ylabel('u(x, t)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'time_profiles_CFL_{mp:.4f}_theta_{theta}.png', dpi=300)

        plt.show()
