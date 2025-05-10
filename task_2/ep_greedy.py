import numpy as np
import random
import matplotlib.pyplot as plt
from collections import Counter

# payoff matrix
A = np.array([[1.0, 0.0],
              [2/3, 2/3]])

B = np.array([[1.0, 2/3],
              [0.0, 2/3]])

Q1 = np.zeros(2)
Q2 = np.zeros(2)

alpha = 0.1
epsilon = 0.1
episodes = 10000  # 改小方便可视化动作序列
runs = 50  # 学习次数

# 记录联合策略选择
joint_choices = []  # 形式为 ('S', 'H'), ('H', 'H') 等

for run in range(runs):
    Q1 = np.zeros(2)
    Q2 = np.zeros(2)
    # Q1 = np.random.rand(2)
    # Q2 = np.random.rand(2)

    for _ in range(episodes):
        # ε-greedy 策略选择动作
        a1 = random.randint(0, 1) if random.random() < epsilon else np.argmax(Q1)
        a2 = random.randint(0, 1) if random.random() < epsilon else np.argmax(Q2)

        r1 = A[a1, a2]
        r2 = B[a1, a2]

        Q1[a1] += alpha * (r1 - Q1[a1])
        Q2[a2] += alpha * (r2 - Q2[a2])

    # 最终策略选择（greedy）
    final_a1 = np.argmax(Q1)
    final_a2 = np.argmax(Q2)
    joint_choices.append((final_a1, final_a2))

# 统计联合分布
joint_counter = Counter(joint_choices)

# 打印结果
print("联合策略分布统计:")
for key in sorted(joint_counter):
    print(f"{key}: {joint_counter[key]}")

# 可视化为热力图矩阵
matrix = np.zeros((2, 2))  # rows: P1(S,H), cols: P2(H,S)
for (a1, a2), count in joint_counter.items():
    matrix[a1, 1 - a2] = count

fig, ax = plt.subplots()
im = ax.imshow(matrix, cmap='Blues')

# 标签
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['H', 'S'])
ax.set_yticklabels(['S', 'H'])
ax.set_xlabel("Player 2")
ax.set_ylabel("Player 1")
ax.set_title("Joint Final Strategy Distribution")

# 显示每个格子的数量
for i in range(2):
    for j in range(2):
        ax.text(j, i, int(matrix[i, j]), ha='center', va='center', color='black')

plt.tight_layout()
plt.show()