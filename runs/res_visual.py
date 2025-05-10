import os
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# === 可配置参数 ===
param_names = ["lr", "clip"]  # 扫描的参数名
base_dir = ""
target_agent = "archer_0"     # 要跟踪的 agent
# ===================

def extract_rewards_from_log(log_path):
    rewards = []
    # 匹配格式：123 {'archer_0': 45.6}
    pattern = re.compile(rf"(\d+)\s+\{{['\"]{target_agent}['\"]:\s*([0-9.]+)\}}")
    with open(log_path, "r", encoding='utf-8') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                iteration = int(match.group(1))
                reward = float(match.group(2))
                rewards.append((iteration, reward))
    return rewards

def parse_param_values(dirname: str):
    # 从目录名中解析出形如 "lr_1e-04__clip_0.2" 的参数组合
    parts = dirname.split("__")
    values = {}
    for part in parts:
        for param in param_names:
            if part.startswith(f"{param}_"):
                values[param] = part[len(param) + 1:]
    return values

def plot_all_runs(param_names, base_dir="runs"):
    base_path = Path(base_dir)
    all_data = {}

    for param_dir in sorted(base_path.iterdir()):
        if not param_dir.is_dir():
            continue

        param_values = parse_param_values(param_dir.name)
        if len(param_values) != len(param_names):
            continue  # skip irrelevant dirs

        label = ", ".join(f"{k}={v}" for k, v in param_values.items())
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
            all_data[label] = run_rewards

    # 绘图
    plt.figure(figsize=(12, 6))
    for label, runs in sorted(all_data.items()):
        min_len = min(len(r) for r in runs)
        trimmed = [r[:min_len] for r in runs]

        rewards_array = np.array(trimmed)
        avg_rewards = np.mean(rewards_array, axis=0)
        std_rewards = np.std(rewards_array, axis=0)
        steps = np.arange(min_len)

        plt.plot(steps, avg_rewards, label=label)
        plt.fill_between(steps, avg_rewards - std_rewards, avg_rewards + std_rewards, alpha=0.2)

    plt.xlabel("Training Iteration")
    plt.ylabel(f"Average Episode Reward ({target_agent})")
    title_params = ", ".join(param_names)
    plt.title(f"Training Performance by {title_params}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig(f"{'_'.join(param_names)}_comparison.png")
    plt.show()

if __name__ == "__main__":
    plot_all_runs(param_names, base_dir)