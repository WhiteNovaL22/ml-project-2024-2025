import numpy as np
import matplotlib.pyplot as plt

# Define the Stag Hunt payoff matrices
A = np.array([[1.0, 0.0],
              [2/3, 2/3]])

B = np.array([[1.0, 2/3],
              [0.0, 2/3]])

# Parameters
alpha = 1.0
tau = 0.1
kappa = 5  # leniency degree

# Define lenient utility function
def lenient_util(payoff_matrix, opponent_policy, kappa):
    u = np.zeros(2)
    for i in range(2):
        total = 0
        for j in range(2):
            a_ij = payoff_matrix[i, j]
            # count p where a_ik <= a_ij
            pk_leq = sum(opponent_policy[k] for k in range(2) if payoff_matrix[i, k] <= a_ij)
            pk_lt  = sum(opponent_policy[k] for k in range(2) if payoff_matrix[i, k] < a_ij)
            pk_eq  = sum(opponent_policy[k] for k in range(2) if payoff_matrix[i, k] == a_ij)

            if pk_eq > 0:
                weight = (pk_leq ** kappa - pk_lt ** kappa) / pk_eq
                total += a_ij * opponent_policy[j] * weight
        u[i] = total
    return u

# Define the vector field function for Lenient Boltzmann Q-learning dynamics
def lenient_boltzmann_vector_field(x1, y1):
    x = np.array([x1, 1 - x1])
    y = np.array([y1, 1 - y1])

    # Lenient utility
    u1 = lenient_util(A, y, kappa)
    avg_u1 = np.dot(x, u1)
    dx1 = (alpha * x1 / tau) * (u1[0] - avg_u1) - alpha * x1 * (np.log(x1 + 1e-10) - np.sum(x * np.log(x + 1e-10)))

    u2 = lenient_util(B.T, x, kappa)
    avg_u2 = np.dot(y, u2)
    dy1 = (alpha * y1 / tau) * (u2[0] - avg_u2) - alpha * y1 * (np.log(y1 + 1e-10) - np.sum(y * np.log(y + 1e-10)))

    return dx1, dy1

# Create grid for vector field
grid_points = 21
X, Y = np.meshgrid(np.linspace(0.01, 0.99, grid_points), np.linspace(0.01, 0.99, grid_points))
U, V = np.zeros_like(X), np.zeros_like(Y)

# Compute the vector field over the grid
for i in range(grid_points):
    for j in range(grid_points):
        dx, dy = lenient_boltzmann_vector_field(X[i, j], Y[i, j])
        U[i, j] = dx
        V[i, j] = dy

# Plot the vector field
plt.figure(figsize=(7, 6))
plt.quiver(X, Y, U, V, angles="xy", scale_units="xy", color="gray", width=0.003)
plt.xlabel("Player 1 chooses S (x1)")
plt.ylabel("Player 2 chooses S (y1)")
plt.title(f"Lenient Boltzmann Q-learning Dynamics (κ = {kappa})")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True)
plt.tight_layout()
plt.show()