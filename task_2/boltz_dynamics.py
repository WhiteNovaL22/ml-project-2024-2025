import numpy as np
import matplotlib.pyplot as plt

# 定义博弈（以 Stag Hunt 为例）
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

alpha = 0.1  # 可以视为比例因子，向量场中不影响方向
tau = 0.1    # Boltzmann 温度参数

# 微分方程右侧：计算 dx1 和 dy1
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

# 绘图
plt.figure(figsize=(7, 6))
plt.quiver(X, Y, U, V, angles="xy", scale_units="xy", color='gray', width=0.003)
plt.xlabel("Player 1 chooses S (x1)")
plt.ylabel("Player 2 chooses S (y1)")
plt.title("Boltzmann Q-learning Dynamics Vector Field")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True)
plt.show()