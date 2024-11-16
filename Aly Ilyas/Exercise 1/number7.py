#Aly Ilyas
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

# Parameters
L = 1.0  # Right boundary
Lp = -1.0  # Left boundary
J_values = [200, 100, 40]  # Number of spatial points
#Delta_x = (L - Lp) / J  # Spatial step size
#x = np.linspace(Lp, L, J + 1)  # Spatial grid points
T = 1  # Total time for simulation
Delta_t = 0.05  # Different time step sizes
theta_values = [0, 0.5, 1]
# Coefficients
epsilon = 0.001 # Diffusion constant
a = 1  # Advection speed
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


# Initial condition (polynomial function)
def initial_condition(x):
    polynomial_sum = np.sum([b_k[k] * phi_k(x) for k, phi_k in enumerate([phi_0, phi_1, phi_2, phi_3])], axis=0)
    return (1 - x)**4 * (1 + x) * (polynomial_sum + C)

def build_matrices(J, epsilon, a, Delta_x, Delta_t, theta):
    nu = Delta_t / Delta_x
    mu = Delta_t / Delta_x**2

    A = np.zeros((J + 1, J + 1))
    B = np.zeros((J + 1, J + 1))

    for j in range(1, J):
        A[j, j] = 1 + (theta * nu * a) + (2 * theta * mu * epsilon)
        A[j, j - 1] = -theta *  mu * epsilon
        A[j, j + 1] = -theta * (nu * a + mu * epsilon)

        # Fill B matrix (coefficients of u_j^n)
        B[j, j] = 1 - ((1 - theta)*nu *a)-((1 - theta)*mu * epsilon)
        B[j, j - 1] = (1-theta) *  mu * epsilon
        B[j, j + 1] = (1-theta) * (nu * a + mu * epsilon)

    # Apply boundary conditions (Dirichlet)
    A[0, 0] = A[-1, -1] = 1
    B[0, 0] = B[-1, -1] = 1

    return A, B

# Time-stepping function using theta-scheme
def theta_scheme(A, B, u, num_steps):
    solutions = [u.copy()]
    for _ in range(num_steps):
        # Solve A u_new = B u_old
        u_new = solve(A, B @ u)
        u_new[0], u_new[-1] = 0, 0  # Apply Dirichlet boundary conditions
        u = u_new.copy()
        solutions.append(u.copy())
    return solutions

# Main simulation
for J in J_values:
    Delta_x = (L - Lp) / J
    print(Delta_x)
    x = np.linspace(Lp, L, J + 1)
    C = compute_C()
    print(C)
    for theta in theta_values:
        u0 = initial_condition(x)
        num_steps = int(T / Delta_t)

        A, B = build_matrices(J, epsilon, a, Delta_x, Delta_t, theta)
        solutions = theta_scheme(A, B, u0, num_steps)
        mp = a*Delta_t / Delta_x
        #mu = Delta_t / Delta_x ** 2
        # Plotting solutions at different time steps
        time_points = [0, int(num_steps / 4), int(2*num_steps / 4), int(3 * num_steps / 4), num_steps]
        # Plot the full time profiles
        plt.figure(figsize=(8, 6))
        for i, step in enumerate(time_points):
            plt.plot(x, solutions[step], label=f't = {step * Delta_t:.2f}s')
        #plt.title(f'Time Profiles for μ = {mu:.4f}, θ = {theta}')
        plt.title(f'Time Profiles for Δx = {Delta_x}, CFL={mp}, θ = {theta}')
        plt.xlabel('x')
        plt.ylabel('u(x,t)')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'7a_CFL_{mp:.4f}_theta_{theta}.png', dpi=300)
        plt.show()
        # Zoom in near x = -1
        plt.figure(figsize=(10, 6))
        for i, step in enumerate(time_points):
            plt.plot(x, solutions[step], label=f't = {step * Delta_t:.2f}s')
        plt.xlim([-1, -0.9])  # Zoom in near x = -1
        #plt.ylim([0, 0.1])
        plt.title(f'Time Profiles near x = -1 for Δx = {Delta_x},  CFL={mp}, θ = {theta}')
        plt.xlabel('x')
        plt.ylabel('u(x, t)')
        plt.legend(loc='best')  # Corrected the title for zoomed-in plots
        plt.grid(True)
        plt.savefig(f'7bdx{Delta_x}e{epsilon}{theta}.png', dpi=300)
        plt.show()