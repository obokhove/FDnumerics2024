#Aly Ilyas
import numpy as np
import matplotlib.pyplot as plt

# Parameters
L_p = -1.0  # Left boundary
L = 1.0  # Right boundary
T = 1  # Total time for simulation
a = 1  # Advection constant
epsilon = 0.001  # Diffusion constant
theta = 1  # Crank-Nicolson scheme (theta = 1)

# Different spatial step sizes for refinement (finer grids)
Delta_x_values = [0.5, 0.05, 0.005]
# Time step size (CFL condition to be checked)
Delta_t = 0.001

# Define Legendre polynomials
def phi_0(x):
    return np.ones_like(x)

def phi_1(x):
    return x

def phi_2(x):
    return (3 / 2) * x ** 2 - 1 / 2

def phi_3(x):
    return (5 / 2) * x ** 3 - (3 / 2) * x

# Random coefficients
np.random.seed(42)  # For reproducibility
b_k = np.random.uniform(0, 1, 4)

# Print the coefficients
print("Random coefficients b_k:", b_k)

# Compute C such that the initial condition is non-negative
def compute_C(x):
    values = np.vstack([phi_0(x), phi_1(x), phi_2(x), phi_3(x)]).T
    return max(-np.sum(b_k * values, axis=1))

# Initial condition
def initial_condition(x, C):
    polynomial_sum = np.sum([b_k[k] * phi_k(x) for k, phi_k in enumerate([phi_0, phi_1, phi_2, phi_3])], axis=0)
    return (1 - x) ** 4 * (1 + x) * (polynomial_sum + C)

# Function to implement the explicit scheme
def explicit_scheme(Delta_x, Delta_t, max_time, theta):
    J = int((L - L_p) / Delta_x)  # Number of spatial points
    x = np.linspace(L_p, L, J + 1)  # Spatial grid points
    mu = epsilon * Delta_t / Delta_x ** 2  # Diffusion coefficient
    gamma = a * Delta_t / Delta_x  # Advection coefficient
    C = compute_C(x)

    u = initial_condition(x, C)
    u_new = np.zeros_like(u)
    time_profiles = []

    current_time = 0.0  # Start from time = 0
    time_intervals = [0, 0.25, 0.5, 0.75, 1]  # Times to store results
    captured_times = set()  # Set to track already captured times

    while current_time <= max_time:
        # Store solution at specific time intervals
        for target_time in time_intervals:
            if np.isclose(current_time, target_time, atol=Delta_t) and target_time not in captured_times:
                time_profiles.append((current_time, u.copy()))
                captured_times.add(target_time)

        # Apply the explicit scheme for internal points
        for j in range(1, J):
            u_new[j] = u[j] + theta * (mu * (u[j - 1] - 2 * u[j] + u[j + 1])) + (1 - theta) * (
                        u[j] + gamma * (u[j + 1] - u[j]))

        # Apply boundary conditions
        u_new[0] = 0  # u(L_p, t) = 0
        u_new[-1] = 0  # u(L, t) = 0

        # Update the solution
        u = u_new.copy()

        # Update the current time
        current_time += Delta_t

    return x, time_profiles

# Plotting the results for different Delta_x values
for Delta_x in Delta_x_values:
    x, time_profiles = explicit_scheme(Delta_x, Delta_t, T, theta)

    # Plot the full time profiles
    plt.figure(figsize=(10, 6))
    for time, profile in time_profiles:
        plt.plot(x, profile, label=f't = {time:.2f}')
    plt.title(f'Time Profiles for Δx = {Delta_x}, and ε = {epsilon}')
    plt.xlabel('x')
    plt.ylabel('u(x, t)')
    plt.legend(loc='best')  # Make sure legend location doesn't overlap
    plt.grid(True)
    plt.savefig(f'time_profiles_fordx{Delta_x}e{epsilon}.png', dpi=300)
    plt.show()

    # Zoom in near x = -1
    plt.figure(figsize=(10, 6))
    for time, profile in time_profiles:
        plt.plot(x, profile, label=f't = {time:.2f}')
    plt.xlim([-1, -0.8])  # Zoom in near x = -1
    plt.title(f'Time Profiles near x = -1 for Δx = {Delta_x}, ε = {epsilon}')
    plt.xlabel('x')
    plt.ylabel('u(x, t)')
    plt.legend(loc='best')  # Corrected the title for zoomed-in plots
    plt.grid(True)
    plt.savefig(f'zoomtime_profiles_fordx{Delta_x}e{epsilon}.png', dpi=300)
    plt.show()
