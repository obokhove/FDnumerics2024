#Aly Ilyas
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

# Parameters
L = 1.0  # Right boundary
Lp = -1.0  # Left boundary
J = 40  # Number of spatial points
Delta_x = (L - Lp) / J  # Spatial step size
print(Delta_x)
x = np.linspace(Lp, L, J + 1)  # Spatial grid points
T = 1  # Total time for simulation
Delta_t = 0.01  # Time step size
theta = 0.5  # 0.5 = Crank-Nicolson

# Coefficients
epsilon = 0 # Diffusion constant
a = 1  # Advection speed

# Initial condition (polynomial function)
def initial_condition(x):
    return (1 - x) ** 4 * (1 + x)

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
u0 = initial_condition(x)
num_steps = int(T / Delta_t)

A, B = build_matrices(J, epsilon, a, Delta_x, Delta_t, theta)
solutions = theta_scheme(A, B, u0, num_steps)

# Plotting solutions at different time steps
time_points = [0, int(num_steps / 4), int(2*num_steps / 4), int(3 * num_steps / 4), num_steps]
plt.figure(figsize=(8, 6))
for i, step in enumerate(time_points):
    plt.plot(x, solutions[step], label=f't = {step * Delta_t:.2f}s')
plt.title(f'Theta-Scheme Advection-Diffusion (θ={theta}, Ɛ={epsilon})')
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.grid(True)
plt.legend()
plt.savefig('5b.png', dpi=300)
plt.show()
