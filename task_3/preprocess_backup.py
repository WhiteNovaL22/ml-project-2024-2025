from pathlib import Path
from typing import Callable

import gymnasium
import pettingzoo
from gymnasium import spaces
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.core.rl_module import RLModule, MultiRLModule
from ray.rllib.core.rl_module import MultiRLModuleSpec, RLModuleSpec
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv, ParallelPettingZooEnv
from ray.rllib.examples.rl_modules.classes.random_rlm import RandomRLModule
from ray.tune.registry import register_env
import numpy as np
import torch

from utils import create_environment

# class CustomWrapper(BaseWrapper):
#     # This is an example of a custom wrapper that flattens the symbolic vector state of the environment
#     # Wrapper are useful to inject state pre-processing or feature that does not need to be learned by the agent
#
#     def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
#         return  spaces.flatten_space(super().observation_space(agent))
#
#     def observe(self, agent: AgentID) -> ObsType | None:
#         obs = super().observe(agent)
#         flat_obs = obs.flatten()
#         return flat_obs

# class CustomWrapper(BaseWrapper):
#     def __init__(self, env, num_archers=1, max_arrows=5, max_zombies=4):
#         super().__init__(env)
#         self.num_archers = num_archers
#         self.max_arrows = max_arrows
#         self.max_zombies = max_zombies
#
#     def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
#         # 注意我们现在只保留部分信息，因此空间要重新定义
#         obs_dim = (1 + self.num_archers + self.max_arrows + 2) * 5
#         return spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
#
#     def observe(self, agent: AgentID) -> ObsType | None:
#         obs = super().observe(agent)  # shape: (N+1, 5)
#         if obs is None:
#             return None
#
#         obs = np.array(obs)
#         idx = 0
#
#         agent_self = obs[idx]                            # 1行
#         idx += 1
#         archers = obs[idx:idx + self.num_archers]        # num_archers行
#         idx += self.num_archers
#         # 跳过骑士和剑，因为你设置的是0个骑士
#         arrows = obs[idx:idx + self.max_arrows]          # max_arrows行
#         idx += self.max_arrows
#         zombies = obs[idx:idx + self.max_zombies]        # max_zombies行
#
#         # 找出最近的2个非零僵尸
#         nonzero_zombies = [z for z in zombies if not np.allclose(z, 0)]
#         nearest_zombies = sorted(nonzero_zombies, key=lambda z: z[0])[:2]
#         # 如果数量不足2个，用全0补齐
#         while len(nearest_zombies) < 2:
#             nearest_zombies.append(np.zeros(5))
#
#         # 拼接我们关心的行：当前agent + 弓箭手 + 箭矢 + 最近2个僵尸
#         selected_rows = [agent_self] + list(archers) + list(arrows) + nearest_zombies
#         flat_obs = np.concatenate(selected_rows).astype(np.float32)
#         return flat_obs

def pairwise_distances(positions):
    positions = np.array(positions)  # shape: (N, 2)
    if len(positions) < 2:
        return np.zeros(4)

    # (N, N, 2)
    diffs = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    # (N, N)
    dists = np.linalg.norm(diffs, axis=-1)

    # 只取上三角（i < j）
    i_upper = np.triu_indices(len(positions), k=1)
    pairwise = dists[i_upper]

    return np.array([
        np.mean(pairwise),
        np.min(pairwise),
        np.max(pairwise),
        np.std(pairwise)
    ])

class CustomWrapper(BaseWrapper):
    def __init__(self, env):
        super().__init__(env)
        raw_env = env.unwrapped
        self.num_archers = raw_env.num_archers
        self.max_arrows = raw_env.max_arrows
        self.max_zombies = raw_env.max_zombies
        self.num_zombies_used = min(4, self.max_zombies)
        # print(self.num_archers, self.max_arrows, self.num_zombies_used)

    def observation_space(self, agent):
        # 原始实体信息：[self] + [archers] + [arrows] + [selected zombies]
        # 每个实体是5维
        base_dim = (1 + self.num_archers + self.max_arrows + self.num_zombies_used) * 5

        # 加入僵尸距离统计特征（4维）+ 聚集中心向量（2维）
        extra_dim = 4 + 2
        total_dim = base_dim + extra_dim

        return spaces.Box(low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32)

    def observe(self, agent):
        obs = super().observe(agent)
        if obs is None:
            return None

        obs = np.array(obs)
        idx = 0
        agent_self = obs[idx]
        idx += 1

        archers = obs[idx:idx + self.num_archers]
        idx += self.num_archers

        # 跳过 knights 和 swords（假设为0个）
        arrows = obs[idx:idx + self.max_arrows]
        idx += self.max_arrows

        zombies = obs[idx:idx + self.max_zombies]

        # ---- 选择最近的4个僵尸 ----
        valid_zombies = [z for z in zombies if not np.allclose(z, 0)]
        valid_zombies = sorted(valid_zombies, key=lambda z: z[0])
        selected_zombies = valid_zombies[:self.num_zombies_used]

        # 补足不足的
        while len(selected_zombies) < self.num_zombies_used:
            selected_zombies.append(np.zeros(5))

        # ---- 提取位置 ----
        zombie_positions = [z[1:3] for z in selected_zombies if not np.allclose(z, 0)]

        # ---- 距离统计特征 ----
        dist_feats = pairwise_distances(zombie_positions)

        # ---- 聚集中心方向与距离 ----
        if zombie_positions:
            zombie_positions = np.array(zombie_positions)
            center = np.mean(zombie_positions, axis=0)  # [rel_x, rel_y]
        else:
            center = np.zeros(2)

        # ---- 拼接 ----
        selected_rows = [agent_self] + list(archers) + list(arrows) + selected_zombies
        flat_obs = np.concatenate(selected_rows + [dist_feats, center])
        return flat_obs.astype(np.float32)