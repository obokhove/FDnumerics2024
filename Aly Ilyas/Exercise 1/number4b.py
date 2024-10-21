#Aly Ilyas
import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 1.0        # Right boundary
Lp = -1.0      # Left boundary
J = 40         # Number of spatial points
Delta_x = (L - Lp) / J  # Spatial step size
x = np.linspace(Lp, L, J + 1)  # Spatial grid points
T = 100.0      # Total time for simulation

# Two different time steps
Delta_t_1 = 0.05   # Stable case
Delta_t_2 = 0.06  # Unstable case

# Coefficients
mu_1 = Delta_t_1 / (Delta_x ** 2)  # Diffusion coefficient for stable case
mu_2 = Delta_t_2 / (Delta_x ** 2)  # Diffusion coefficient for unstable case
gamma_1 = Delta_t_1 / Delta_x       # Advection coefficient for stable case
gamma_2 = Delta_t_2 / Delta_x       # Advection coefficient for unstable case
epsilon = 1e-3                      # Diffusion constant
a = 1.0                             # Advection speed

# Initial condition (polynomial function)
def initial_condition(x):
    u0 = np.zeros_like(x)
    for j in range(len(x)):
        u0[j] = ((1 - (Lp + (j * Delta_x))) ** 4) * (1 + (Lp + (j * Delta_x)))  # Modify as needed
    return u0

# Function to implement the explicit scheme
def explicit_scheme(mu, gamma, Delta_t, max_steps, time_steps_to_store):
    u = initial_condition(x)
    u_new = np.zeros_like(u)
    time_profiles = []

    for n in range(max_steps):
        # Store solution at specific steps
        if n in time_steps_to_store:
            time_profiles.append(u.copy())

        # Apply the explicit scheme for internal points
        for j in range(1, J):
            u_new[j] = (epsilon * mu * u[j-1] +
                         (1 - 2 * epsilon * mu - a * gamma) * u[j] +
                         (a * gamma + epsilon * mu) * u[j+1])

        # Apply boundary conditions
        u_new[0] = 0  # u(Lp, t) = 0
        u_new[-1] = 0  # u(L, t) = 0

        # Update the solution
        u = u_new.copy()

    return time_profiles

# Run the explicit scheme for both time steps
max_steps_1 = int(T / Delta_t_1)
max_steps_2 = int(T / Delta_t_2)
time_steps_1 = [int(t / Delta_t_1) for t in range(0, 5, 1)]  # time steps for Delta_t_1
time_steps_2 = [int(t / Delta_t_2) for t in range(0, 5, 1)]  # time steps for Delta_t_2

# Get the time profiles for both stable and unstable cases
time_profiles_1 = explicit_scheme(mu_1, gamma_1, Delta_t_1, max_steps_1, time_steps_1)
time_profiles_2 = explicit_scheme(mu_2, gamma_2, Delta_t_2, max_steps_2, time_steps_2)

time_step_labels = [f't = {t}' for t in range(0, 5, 1)]
plt.figure()
plt.title(f'Stable case (Δt = {Delta_t_1})')
for idx, profile in enumerate(time_profiles_1):
    plt.plot(x, profile, label=f'{time_step_labels[idx]}')
plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.savefig('stable_case_advection_diffusion.png')
plt.show()
plt.figure()
plt.title(f'Unstable case (Δt = {Delta_t_2})')
for idx, profile in enumerate(time_profiles_2):
    plt.plot(x, profile, label=f'{time_step_labels[idx]}')
plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.savefig('unstable_case_advection_diffusion.png', dpi=300)
plt.show()


