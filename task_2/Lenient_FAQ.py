import numpy as np
import matplotlib.pyplot as plt

# payoff matrix
A = np.array([[1.0, 0.0],
              [2/3, 2/3]])
B = np.array([[1.0, 2/3],
              [0.0, 2/3]])

# A = np.array([[3.0, 0.0],
#               [5.0, 1.0]])
#
# B = np.array([[3.0, 5.0],
#               [0.0, 1.0]])

# A = np.array([[0.0, 1.0],
#               [1.0, 0.0]])
#
# B = np.array([[1.0, 0.0],
#               [0.0, 1.0]])

# A = np.array([[12.0, 0.0],
#               [11.0, 10.0]])
#
# B = np.array([[12.0, 11.0],
#               [0.0, 10.0]])

# Lenient Boltzmann Q-learning 参数
alpha = 0.005           # 学习率
tau = 0.1             # Boltzmann 温度（控制探索程度）
kappa = 25             # 包容性采样次数
episodes = 100000       # 总训练轮数

# 创建对称分布的起始点（比如沿对角线）
start_points = [(0.3, 0.7), (0.7, 0.3),
                (0.25, 0.25), (0.5, 0.5), (0.75, 0.75),
                (0.7, 0.9), (0.9, 0.7),
                (0.1, 0.8), (0.8, 0.1),]

# 初始化 Q 表（两个智能体）
Q1 = np.zeros(2)
Q2 = np.zeros(2)

strategy_history = []

# 每个动作的奖励缓存（用于 leniency）
r1_buffer = {0: [], 1: []}
r2_buffer = {0: [], 1: []}

plt.figure(figsize=(8, 8))

# Boltzmann softmax
def boltzmann(Q_values, tau):
    Q_values = Q_values - np.max(Q_values)  # 防止 overflow
    exp_Q = np.exp(Q_values / tau)
    return exp_Q / np.sum(exp_Q)

# Define lenient utility function
def lenient_util(payoff_matrix, opponent_policy, kappa):
    u = np.zeros(2)
    for i in range(2):
        total = 0
        for j in range(2):
            a_ij = payoff_matrix[i, j]
            pk_leq = sum(opponent_policy[k] for k in range(2) if payoff_matrix[i, k] <= a_ij) # count p where a_ik <= a_ij
            pk_lt  = sum(opponent_policy[k] for k in range(2) if payoff_matrix[i, k] < a_ij) # count p where a_ik < a_ij
            pk_eq  = sum(opponent_policy[k] for k in range(2) if payoff_matrix[i, k] == a_ij) # count p where a_ik == a_ij

            if pk_eq > 0: # pk_eq != 0, !< 0
                weight = (pk_leq ** kappa - pk_lt ** kappa) / pk_eq
                total += a_ij * opponent_policy[j] * weight
        u[i] = total
    return u

# Define the vector field function for Lenient Boltzmann Q-learning dynamics
def lenient_boltzmann_dynamics(x1, y1):
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


for p1_start, p2_start in start_points:
    # 构造使策略概率为指定起点的 Q 值
    Q1 = np.array([tau * np.log(p1_start), tau * np.log(1 - p1_start)])
    Q2 = np.array([tau * np.log(p2_start), tau * np.log(1 - p2_start)])

    strategy_history = []

    # 训练主循环
    for episode in range(episodes):
        # 根据当前 Q 值通过 Boltzmann 策略选择动作
        p1_probs = boltzmann(Q1, tau)
        p2_probs = boltzmann(Q2, tau)
        a1 = np.random.choice([0, 1], p=p1_probs)
        a2 = np.random.choice([0, 1], p=p2_probs)

        strategy_history.append((p1_probs[0], p2_probs[0]))

        # 获取即时回报
        r1 = A[a1, a2]
        r2 = B[a1, a2]

        # 存储回报，用于宽容更新
        r1_buffer[a1].append(r1)
        r2_buffer[a2].append(r2)

        # 满足 κ 次后执行基于最大回报的 Q 值更新
        if len(r1_buffer[a1]) >= kappa:
            Q1[a1] += (alpha / (p1_probs[a1] + 1e-10)) * (max(r1_buffer[a1]) - Q1[a1])
            r1_buffer[a1] = []

        if len(r2_buffer[a2]) >= kappa:
            Q2[a2] += (alpha / (p2_probs[a2] + 1e-10)) * (max(r2_buffer[a2]) - Q2[a2])
            r2_buffer[a2] = []

    traj = np.array(strategy_history)

    # 起点标记
    plt.scatter(p1_start, p2_start, s=20, zorder=5)
    # 收敛点标记
    plt.scatter(traj[-1, 0], traj[-1, 1], s=20, c='black', marker='x', zorder=6)

    plt.plot(traj[:, 0], traj[:, 1])

# Create grid for vector field
grid_points = 20
X, Y = np.meshgrid(np.linspace(0.01, 0.99, grid_points), np.linspace(0.01, 0.99, grid_points))
U, V = np.zeros_like(X), np.zeros_like(Y)

# Compute the vector field over the grid
for i in range(grid_points):
    for j in range(grid_points):
        dx, dy = lenient_boltzmann_dynamics(X[i, j], Y[i, j])
        U[i, j] = dx
        V[i, j] = dy

# Plot the vector field
plt.quiver(X, Y, U, V, angles="xy", scale_units="xy", color="lightgray")
plt.xlabel("Player 1 chooses S (x1)")
plt.ylabel("Player 2 chooses S (y1)")
plt.title(f"Lenient FAQ-learning Strategy Evolution (κ = {kappa})")
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.grid(True)
plt.tight_layout()
plt.show()