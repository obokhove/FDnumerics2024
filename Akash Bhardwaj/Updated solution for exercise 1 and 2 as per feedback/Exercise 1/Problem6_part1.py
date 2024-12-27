import numpy as np
import matplotlib.pyplot as plt


Lp = -1  
L = 1  
u_x0 = 0  
u_x1 = 0  
T_max = 1  # Maximum simulation time
J = 40  # Number of grid points
delta_x = (L - Lp) / float(J)
x_values = [Lp + j * delta_x for j in range(J + 1)]
a = 1  # Advection coefficient
epsilon = 1e-3  # Diffusion coefficient

# Initial condition
def initial_condition(x):
    return ((1 - x)**4) * (1 + x)

# Implementation of the θ-method
def Un1_theta_method(Un_vec, delta_t, delta_x, theta=0.5, a=1, epsilon=1e-3):
    m = len(Un_vec) - 2
    A = np.zeros((m, m))
    b = np.zeros((m, 1))
    mu = delta_t / (delta_x * delta_x)
    nu = delta_t / delta_x
    # First row (j = 1)
    A[0, 0] = 1.0 + (theta * nu * a) + (2.0 * theta * epsilon * mu)
    A[0, 1] = -theta * (nu * a + epsilon * mu)

    b[0] = (
        (1 - (1 - theta) * (nu * a + 2.0 * epsilon * mu)) * Un_vec[1]
        + (1 - theta) * (epsilon * mu) * Un_vec[0]
        + (1 - theta) * (nu * a + epsilon * mu) * Un_vec[2]
    )

    # Middle rows (j = 2 to m-1)
    for k in range(1, m - 1):
        j = k + 1
        A[k, k] = 1.0 + (theta * nu * a) + (2.0 * theta * epsilon * mu)
        A[k, k + 1] = -theta * (nu * a + epsilon * mu)
        A[k, k - 1] = -theta * epsilon * mu

        b[k] = (
            (1 - (1 - theta) * (nu * a + 2.0 * epsilon * mu)) * Un_vec[j]
            + (1 - theta) * (epsilon * mu) * Un_vec[j - 1]
            + (1 - theta) * (nu * a + epsilon * mu) * Un_vec[j + 1]
        )

    # Last row (j = m)
    A[m - 1, m - 1] = 1.0 + (theta * nu * a) + (2.0 * theta * epsilon * mu)
    A[m - 1, m - 2] = -theta * epsilon * mu

    j = m
    b[m - 1] = (
        (1 - (1 - theta) * (nu * a + 2.0 * epsilon * mu)) * Un_vec[j]
        + (1 - theta) * (epsilon * mu) * Un_vec[j - 1]
        + (1 - theta) * (nu * a + epsilon * mu) * Un_vec[j + 1]
    )

    # Solve the linear system
    Un1 = np.linalg.solve(A, b).flatten()
    Un1 = np.insert(Un1, 0, 0)
    Un1 = np.append(Un1, 0)

    return Un1

# Plotting
h_plot = 3
v_plot = 3
fig, ax = plt.subplots(v_plot, h_plot, sharex=True, sharey=True, figsize=(12, 10))
theta_vec = [0, 0.5, 1.0]
mu_vec = [0.1, 0.5, 0.9]
max_time_vec = [0.25, 0.5, 1]

# Loop for μ and θ combinations
for w, mu in enumerate(mu_vec):
    for i, theta in enumerate(theta_vec):
        delta_t = mu * delta_x * delta_x
        nu = mu * delta_x
        # Initialize solution
        t = 0
        Un = [initial_condition(x) for x in x_values]

        for t_lim in max_time_vec:
            while t < t_lim:
                t += delta_t
                Un1 = Un1_theta_method(Un, delta_t, delta_x, theta=theta, a=a, epsilon=epsilon)
                Un = Un1
            # Plot the solution
            ax[w, i].plot(
                x_values,
                Un1,
                label=f"t = {t:.2f}",
                linestyle="--",
                marker="o",
            )
        ax[w, i].set_title(r"$\mu = $" + f"{mu}, " + r"$\theta = $" + f"{theta}, " + r"$J = $" + f"{J}")
        ax[w, i].legend(loc="upper right")
        if w == v_plot - 1:
            ax[w, i].set_xlabel(r"$x$")
        if i == 0:
            ax[w, i].set_ylabel(r"$U_j^{n}$")

plt.tight_layout()
plt.savefig("Problem6_part1.png", dpi=300, bbox_inches="tight")
plt.show()
