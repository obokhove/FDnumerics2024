#Aly Ilyas
import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 1.0        # Right boundary
Lp = -1.0      # Left boundary
J = 40         # Number of spatial points
Delta_x = (L - Lp) / J  # Spatial step size
print(Delta_x)
x = np.linspace(Lp, L, J + 1)  # Spatial grid points
T = 1.0      # Total time for simulation

# Two different time steps
Delta_t_1 = 0.05   # Stable case
Delta_t_2 = 0.055  # Unstable case
# Coefficients
mu_1 = Delta_t_1 / (Delta_x ** 2)  # Diffusion coefficient for stable case
mu_2 = Delta_t_2 / (Delta_x ** 2)  # Diffusion coefficient for unstable case
nu_1 = Delta_t_1 / Delta_x       # Advection coefficient for stable case
nu_2 = Delta_t_2 / Delta_x       # Advection coefficient for unstable case
epsilon = 0.001                     # Diffusion constant
a = 1.0                             # Advection speed

# Initial condition (polynomial function)
def initial_condition(x):
    return (1 - x) ** 4 * (1 + x)

# Function to implement the explicit scheme
def explicit_scheme(mu, nu, Delta_t, max_steps, time_steps_to_store):
    u = initial_condition(x)
    u_new = np.zeros_like(u)
    time_profiles = []

    for n in range(max_steps):
        # Store solution at specific steps
        if n in time_steps_to_store or n == max_steps - 1:
            time_profiles.append(u.copy())

        # Apply the explicit scheme for internal points
        theta = 0
        for j in range(1, J):
            u_new[j] = (1 - ((1 - theta)*nu *a)-((1 - theta)*mu * epsilon) * u[j-1] +
                        ((1-theta) *  mu * epsilon) * u[j] +
                         ((1-theta) * (nu * a + mu * epsilon)) * u[j+1])

        # Apply boundary conditions
        u_new[0] = 0  # u(Lp, t) = 0
        u_new[-1] = 0  # u(L, t) = 0

        # Update the solution
        u = u_new.copy()

    return time_profiles

# Run the explicit scheme for both time steps

num_steps_1 = int(T / Delta_t_1)
num_steps_2 = int(T / Delta_t_2)
time_steps_1 = [0, int(num_steps_1 / 4), int(2*num_steps_1 / 4), int(3 * num_steps_1 / 4), num_steps_1]
time_steps_2 = [0, int(num_steps_2 / 4), int(2*num_steps_2 / 4), int(3 * num_steps_2 / 4), num_steps_2]

# Get the time profiles for both stable and unstable cases
solution_1 = explicit_scheme(mu_1, nu_1, Delta_t_1, num_steps_1, time_steps_1)
solution_2 = explicit_scheme(mu_2, nu_2, Delta_t_2, num_steps_2, time_steps_2)
# Correct time_step_labels for both cases
time_step_labels_stable = [f't = {t*Delta_t_1:.2f}s' for t in time_steps_1]
time_step_labels_unstable = [f't = {t*Delta_t_2:.2f}s' for t in time_steps_2]

# Plot stable case
plt.figure()
plt.title(f'Stable case (CFL= {a*nu_1:.2f})')
for idx, profile in enumerate(solution_1):
    plt.plot(x, profile, label=f'{time_step_labels_stable[idx]}.2')
plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.savefig('4c.png')
plt.show()

# Plot unstable case
plt.figure()
plt.title(f'Unstable case (CPL = {a*nu_2:.2f})')
for idx, profile in enumerate(solution_2):
    plt.plot(x, profile, label=f'{time_step_labels_unstable[idx]}.2')
plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.savefig('4d.png')
plt.show()



