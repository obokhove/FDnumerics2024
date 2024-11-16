#Aly Ilyas

import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 1.0        # Right boundary
J = 20         # Number of spatial points
Delta_x = L / J  # Spatial step size
x = np.linspace(0, L, J + 1)  # Spatial grid points
T = 0.1        # Total time for simulation

# Two different time steps
Delta_t_1 = 0.0012  # Stable case
Delta_t_2 = 0.0013  # Unstable case

# Diffusion coefficient for heat equation
mu_1 = Delta_t_1 / (Delta_x**2)
mu_2 = Delta_t_2 / (Delta_x**2)

# Initial condition (hat function)
def initial_condition(x):
    u0 = np.zeros_like(x)
    for j in range(len(x)):
        if x[j] <= 0.5:
            u0[j] = 2 * x[j]
        else:
            u0[j] = 2 * (1 - x[j])
    return u0

# Function to implement the explicit scheme
def explicit_scheme(mu, Delta_t, max_steps):
    u = initial_condition(x)
    u_new = np.zeros_like(u)
    time_profiles = []

    for n in range(max_steps):
        # Store solution at specific steps (t = 0, 1, 25, 50)
        if n in [0, 1, 25, 50]:
            time_profiles.append(u.copy())

        # Apply the explicit scheme for internal points
        for j in range(1, J):
            u_new[j] = u[j] + mu * (u[j+1] - 2*u[j] + u[j-1])

        # Apply boundary conditions
        u_new[0] = 0
        u_new[-1] = 0

        # Update the solution
        u = u_new.copy()

    return time_profiles

# Run the explicit scheme for both time steps
max_steps_1 = int(T / Delta_t_1)
max_steps_2 = int(T / Delta_t_2)

time_profiles_1 = explicit_scheme(mu_1, Delta_t_1, max_steps_1)
time_profiles_2 = explicit_scheme(mu_2, Delta_t_2, max_steps_2)

# Labels for the time steps of interest
time_step_labels = ['t = 0', 't = 1', 't = 25', 't = 50']

plt.figure()
plt.title('Stable case (Δt = 0.0012)')
for idx, profile in enumerate(time_profiles_1):
    plt.plot(x, profile, label=f'{time_step_labels[idx]}')
plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.savefig('4a.png')
plt.show()

plt.figure()
plt.title('Unstable case (Δt = 0.0013)')
for idx, profile in enumerate(time_profiles_2):
    plt.plot(x, profile, label=f'{time_step_labels[idx]}')
plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.savefig('4b.png', dpi=300)
plt.show()

