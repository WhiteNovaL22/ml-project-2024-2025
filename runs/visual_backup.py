import os
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# === 修改这里即可 ===
param_name = "lr"  # 可以改为 "gamma", "entropy", "batch_size" 等等
base_dir = "runs"  # 主目录
# =====================

def extract_rewards_from_log(log_path):
    rewards = []
    # pattern = re.compile(r"(\d+).*?archer_0[\"']?\s*:\s*([-\d\.]+)")
    pattern = re.compile(r"(\d+)\s+\{['\"]archer_0['\"]:\s*([0-9.]+)\}")
    with open(log_path, "r", encoding='utf-8') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                iteration = int(match.group(1))
                reward = float(match.group(2))
                rewards.append((iteration, reward))
    return rewards

def plot_all_runs(param_name, base_dir="runs"):
    base_path = Path(base_dir)
    all_data = defaultdict(list)

    # 匹配目录名如 "lr_1e-4"、"gamma_0.99"
    pattern = re.compile(rf"{re.escape(param_name)}_(.+)")

    for param_dir in sorted(base_path.glob(f"{param_name}_*")):
        match = pattern.match(param_dir.name)
        if not match:
            continue
        param_value = match.group(1)
        run_rewards = []

        for run_dir in param_dir.glob("run_*"):
            log_file = run_dir / "log.txt"
            if not log_file.exists():
                continue
            rewards = extract_rewards_from_log(log_file)
            if rewards:
                reward_values = [r for _, r in rewards]
                run_rewards.append(reward_values)

        if run_rewards:
            all_data[param_value] = run_rewards

    # 绘图
    plt.figure(figsize=(12, 6))
    for value, runs in sorted(all_data.items(), key=lambda x: float(x[0].replace("e", "E"))):
        min_len = min(len(r) for r in runs)
        trimmed = [r[:min_len] for r in runs]

        rewards_array = np.array(trimmed)
        avg_rewards = np.mean(rewards_array, axis=0)
        std_rewards = np.std(rewards_array, axis=0)
        steps = np.arange(min_len)

        plt.plot(steps, avg_rewards, label=f"{param_name}={value}")
        plt.fill_between(steps, avg_rewards - std_rewards, avg_rewards + std_rewards, alpha=0.2)

if __name__ == "__main__":
    plot_all_runs(param_name, base_dir)