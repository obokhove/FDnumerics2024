
#### Question 6 part 2

import numpy as np
import matplotlib.pyplot as plt

def legendre_polynomials(x):
    phi_0 = 1
    phi_1 = x
    phi_2 = 3 / 2 * x**2 - 1 / 2
    phi_3 = 5 / 2 * x**3 - 3 / 2 * x
    return [phi_0, phi_1, phi_2, phi_3]
def generate_initial_condition(x_values):
    bk = np.random.uniform(0, 1, 4)  
    phi_values = np.array([legendre_polynomials(x) for x in x_values])  
    base_term = np.sum(bk * phi_values, axis=1)
    C = max(-np.min(base_term), 0)  
    return lambda x: ((1 - x)**4 * (1 + x)) * (np.sum(bk * np.array(legendre_polynomials(x))) + C)
Lp = -1
L = 1
epsilon = 1e-3
T_max = 1
a = 1

# grid
J = 40
delta_x = (L - Lp) / float(J)
x_values = [Lp + j * delta_x for j in range(J + 1)]

## Implementation of the θ-method
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
    Un1 = np.linalg.solve(A, b).flatten()
    Un1 = np.insert(Un1, 0, 0)
    Un1 = np.append(Un1, 0)

    return Un1

# Values for θ and μ
theta_vec = [0, 0.5, 1.0]
mu_vec = [0.49, 0.5, 1.0]
max_time_vec = [0.25, 0.5, 1]

# Plotting
fig, ax = plt.subplots(len(mu_vec), len(theta_vec), sharex=True, sharey=True, figsize=(12, 10))

for w, mu in enumerate(mu_vec):
    for i, theta in enumerate(theta_vec):
        delta_t = mu * delta_x**2
        x_values = [Lp + j * delta_x for j in range(J + 1)]

        # Generate initial condition
        initial_condition_func = generate_initial_condition(x_values)
        Un = [initial_condition_func(x) for x in x_values]

        t = 0
        results = []
        times = []

        for t_lim in max_time_vec:
            while t < t_lim:
                t += delta_t
                Un1 = Un1_theta_method(Un, delta_t, delta_x, theta=theta, a=a, epsilon=epsilon)
                Un = Un1
            results.append(Un)
            times.append(t)

        # Plot results
        for idx, time in enumerate(times):
            ax[w, i].plot(x_values, results[idx], label=f"t = {time:.2f}")
        ax[w, i].set_title(r"$\mu = $" + f"{mu}, " + r"$\theta = $" + f"{theta}, " + r"$J = $" + f"{J}")
        ax[w, i].legend(loc="upper right")

for i in range(len(theta_vec)):
    ax[-1, i].set_xlabel(r"$x$")
for w in range(len(mu_vec)):
    ax[w, 0].set_ylabel(r"$u(x, t)$")

plt.tight_layout()
plt.savefig("Problme6_part2.png", dpi=300)
plt.show()
