import re
import os
import matplotlib.pyplot as plt
import numpy as np

# 固定目录路径
base_dir = r'E:\Work\KUL\Sem_2\Machine_Learning_Project\Project\ml-project-2024-2025\task_3\data'

# 要处理的多个设置文件
filenames = [
    'res_def_1024.txt',
    'res_clip_0_2_1024.txt',
    'res_gae_1024.txt',
    'res_gae_clip_vf.txt'
]

# 每个设置下有多少次实验（你说是5）
num_runs_per_setting = 5

# 提取某个文件中所有 runs 的 reward 数据
def read_multiple_runs(file_path, num_runs):
    all_rewards = []
    current_rewards = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            match = re.match(r"(\d+)\s+\{['\"]archer_0['\"]:\s*([0-9.]+)\}", line)
            if match:
                reward = float(match.group(2))
                current_rewards.append(reward)
            elif 'checkpoint' in line and current_rewards:
                # 分隔多个 run 的信号（也可以用别的判断依据）
                continue

        # 假设每 run 的长度相等，按平均分组
        total = len(current_rewards)
        steps_per_run = total // num_runs
        for i in range(num_runs):
            run_rewards = current_rewards[i * steps_per_run : (i + 1) * steps_per_run]
            all_rewards.append(run_rewards)

    return np.array(all_rewards)  # shape: [num_runs, steps]

# 画平均 reward 曲线
def plot_average_rewards(files):
    plt.figure(figsize=(12, 6))

    for filename in files:
        file_path = os.path.join(base_dir, filename)
        all_runs = read_multiple_runs(file_path, num_runs_per_setting)

        avg_rewards = np.mean(all_runs, axis=0)
        std_rewards = np.std(all_runs, axis=0)
        steps = list(range(len(avg_rewards)))

        plt.plot(steps, avg_rewards, label=f'{filename} (avg)')
        plt.fill_between(steps, avg_rewards - std_rewards, avg_rewards + std_rewards, alpha=0.2)

    plt.xlabel('Iteration')
    plt.ylabel('Average Reward (archer_0)')
    plt.title('Average Reward over Iterations (Multiple Runs)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 执行
plot_average_rewards(filenames)