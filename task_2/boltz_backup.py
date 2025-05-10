import numpy as np
import matplotlib.pyplot as plt

# 固定参数
alpha = 0.1
gamma = 0.9
tau = 0.1
episodes = 8000

# 收益矩阵
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

# Boltzmann softmax
def boltzmann_probs(Q_values, tau):
    Q_values = Q_values - np.max(Q_values)  # 防止 overflow
    exp_Q = np.exp(Q_values / tau)
    return exp_Q / np.sum(exp_Q)

# 计算Boltzmann动力学
def boltzmann_q_vector_field(x1, y1):
    x = np.array([x1, 1 - x1])
    y = np.array([y1, 1 - y1]) # x, y分别为两个玩家的策略（概率）

    # 玩家1的导数
    payoff1 = A @ y # 矩阵乘法
    avg1 = x @ A @ y
    dx1 = x1 * (1 / tau * (payoff1[0] - avg1) - np.log(x1 + 1e-10) + np.sum(x * np.log(x + 1e-10)))

    # 玩家2的导数
    payoff2 = B.T @ x # 转置
    avg2 = x @ B @ y
    dy1 = y1 * (1 / tau * (payoff2[0] - avg2) - np.log(y1 + 1e-10) + np.sum(y * np.log(y + 1e-10)))

    return dx1, dy1

# 创建对称分布的起始点（比如沿对角线）
start_points = [(0.3, 0.7), (0.7, 0.3),
                (0.5, 0.5),
                (0.7, 0.9), (0.9, 0.7),
                (0.1, 0.8), (0.8, 0.1),]

plt.figure(figsize=(8, 8))

for p1_start, p2_start in start_points:
    # 构造使策略概率为指定起点的 Q 值
    Q1 = np.array([[tau * np.log(p1_start), 0], [tau * np.log(1 - p1_start), 0]])
    Q2 = np.array([[tau * np.log(p2_start), tau * np.log(1 - p2_start)], [0, 0]])

    strategy_history = []

    for _ in range(episodes):
        p1_probs = boltzmann_probs(Q1.sum(axis=1), tau)
        p2_probs = boltzmann_probs(Q2.sum(axis=0), tau)

        strategy_history.append((p1_probs[0], p2_probs[0]))

        a1 = np.random.choice([0, 1], p=p1_probs)
        a2 = np.random.choice([0, 1], p=p2_probs)

        r1 = A[a1, a2]
        r2 = B[a1, a2]

        Q1[a1, a2] += alpha * (r1 + gamma * np.max(Q1[:, a2]) - Q1[a1, a2])
        Q2[a1, a2] += alpha * (r2 + gamma * np.max(Q2[a1, :]) - Q2[a1, a2])

    # 起点标记
    plt.scatter(p1_start, p2_start, s=20, zorder=5)

    traj = np.array(strategy_history)
    plt.plot(traj[:, 0], traj[:, 1], lw=1)


# 生成网格
grid_points = 20
X, Y = np.meshgrid(np.linspace(0.01, 0.99, grid_points), np.linspace(0.01, 0.99, grid_points))
U = np.zeros_like(X)
V = np.zeros_like(Y)

# 计算向量
for i in range(grid_points):
    for j in range(grid_points):
        dx, dy = boltzmann_q_vector_field(X[i, j], Y[i, j])
        U[i, j] = dx
        V[i, j] = dy

plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.quiver(X, Y, U, V, angles="xy", scale_units="xy", color='gray', width=0.003)
plt.xlabel("Player 1 chooses S (x1)")
plt.ylabel("Player 2 chooses S (y1)")
plt.title("Boltzmann Q-learning Strategy Evolution")
plt.grid(True)
plt.show()
