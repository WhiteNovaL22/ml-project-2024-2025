import re
import os
import matplotlib.pyplot as plt

base_dir = r'E:\Work\KUL\Sem_2\Machine_Learning_Project\Project\ml-project-2024-2025\task_3\data'

# 修改这里的文件名来读取不同文件
# 要加载的多个日志文件
filenames = [
    'res_def_1024.txt',
    'res_clip_0_2_1024.txt'
]

def read_rewards_from_file(file_path):
    iterations = []
    rewards = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            match = re.match(r"(\d+)\s+\{['\"]archer_0['\"]:\s*([0-9.]+)\}", line)
            if match:
                iter_num = int(match.group(1))
                reward = float(match.group(2))
                iterations.append(iter_num)
                rewards.append(reward)

    return iterations, rewards

# 画出多个文件的 reward 曲线
def plot_multiple_rewards(file_list):
    plt.figure(figsize=(12, 6))

    for filename in file_list:
        file_path = os.path.join(base_dir, filename)
        iterations, rewards = read_rewards_from_file(file_path)
        plt.plot(iterations, rewards, label=filename)

    plt.xlabel('Iteration')
    plt.ylabel('Reward (archer_0)')
    plt.title('Reward over Iterations (Multiple Runs)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 执行函数
plot_multiple_rewards(filenames)