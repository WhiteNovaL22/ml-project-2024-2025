from pathlib import Path
from typing import Optional

import gymnasium
import torch
from gymnasium import spaces
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType
from ray.rllib.core.rl_module import MultiRLModule


# class CustomWrapper(BaseWrapper):
#     """An example of a custom wrapper that flattens the symbolic vector state of the environment.
#
#     Wrappers are useful to inject state pre-processing or features that do not need to be learned by the agent.
#
#     Pay attention to submit the same (or consistent) wrapper you used during training.
#     """
#
#     def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
#         return spaces.flatten_space(super().observation_space(agent))
#
#     def observe(self, agent: AgentID) -> Optional[ObsType]:
#         obs = super().observe(agent)
#         flat_obs = obs.flatten()
#         return flat_obs

# 2 观察最近三个僵尸
class CustomWrapper(BaseWrapper):
    def __init__(self, env):
        super().__init__(env)
        raw_env = env.unwrapped
        self.num_archers = raw_env.num_archers
        self.max_arrows = raw_env.max_arrows
        self.num_knights = raw_env.num_knights
        self.num_swords = self.num_knights
        self.max_zombies = raw_env.max_zombies
        self.num_zombies_used = min(3, self.max_zombies)

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        # 保留 archers + arrows + 3个僵尸（每个5维）
        obs_dim = (self.num_archers + self.max_arrows + self.num_zombies_used) * 5
        return spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def observe(self, agent: AgentID) -> ObsType | None:
        obs = super().observe(agent)  # shape: (N+1, 5)
        if obs is None:
            return None

        obs = np.array(obs)
        idx = 1 # 跳过第一行：当前agent

        archers = obs[idx:idx + self.num_archers]        # 弓箭手
        idx += self.num_archers

        # 跳过 knights 和 swords
        idx += self.num_knights
        idx += self.num_swords

        arrows = obs[idx:idx + self.max_arrows]          # 箭矢
        idx += self.max_arrows

        zombies = obs[idx:idx + self.max_zombies]        # 僵尸

        # 找出最近的最多 3 个僵尸
        nonzero_zombies = [z for z in zombies if not np.allclose(z, 0)]
        nearest_zombies = sorted(nonzero_zombies, key=lambda z: z[0])[:self.num_zombies_used]
        # 补齐不足的
        while len(nearest_zombies) < self.num_zombies_used:
            nearest_zombies.append(np.zeros(5))

        # 拼接：agent + archers + arrows + 3个僵尸
        selected_rows = list(archers) + list(arrows) + nearest_zombies
        flat_obs = np.concatenate(selected_rows).astype(np.float32)
        return flat_obs

# 3 最近4个僵尸 + 分布特征
# def pairwise_distances(positions):
#     positions = np.array(positions)  # shape: (N, 2)
#     if len(positions) < 2:
#         return np.zeros(2)
#
#     # (N, N, 2)
#     diffs = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
#     # (N, N)
#     dists = np.linalg.norm(diffs, axis=-1)
#
#     # 只取上三角（i < j）
#     i_upper = np.triu_indices(len(positions), k=1)
#     pairwise = dists[i_upper]
#
#     return np.array([
#         np.mean(pairwise),
#         np.std(pairwise)
#     ])
#
# class CustomWrapper(BaseWrapper):
#     def __init__(self, env):
#         super().__init__(env)
#         raw_env = env.unwrapped
#         self.num_archers = raw_env.num_archers
#         self.max_arrows = raw_env.max_arrows
#         self.num_knights = raw_env.num_knights
#         self.num_swords = self.num_knights
#         self.max_zombies = raw_env.max_zombies
#         self.num_zombies_used = min(4, self.max_zombies)
#         # print(self.num_archers, self.max_arrows, self.num_knights, self.num_swords, self.num_zombies_used)
#
#     def observation_space(self, agent):
#         # 原始实体信息：[archers] + [arrows] + [selected zombies] -> 跳过弓箭手（自己）
#         # 每个实体是5维
#         base_dim = (self.num_archers + self.max_arrows + self.num_zombies_used) * 5
#
#         # 加入僵尸距离统计特征（2维）+ 聚集中心向量（2维）
#         extra_dim = 2 + 2
#         total_dim = base_dim + extra_dim
#
#         return spaces.Box(low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32)
#
#     def observe(self, agent):
#         obs = super().observe(agent)
#         if obs is None:
#             return None
#
#         obs = np.array(obs)
#         idx = 1 # 跳过自己
#
#         archers = obs[idx:idx + self.num_archers]  # 弓箭手
#         idx += self.num_archers
#
#         # 跳过 knights 和 swords，即使它们存在
#         idx += self.num_knights
#         idx += self.num_swords
#
#         arrows = obs[idx:idx + self.max_arrows]
#         idx += self.max_arrows
#
#         zombies = obs[idx:idx + self.max_zombies]
#
#         # ---- 选择最近的4个僵尸 ----
#         valid_zombies = [z for z in zombies if not np.allclose(z, 0)]
#         valid_zombies = sorted(valid_zombies, key=lambda z: z[0])
#         selected_zombies = valid_zombies[:self.num_zombies_used]
#
#         # 补足不足的
#         while len(selected_zombies) < self.num_zombies_used:
#             selected_zombies.append(np.zeros(5))
#
#         # ---- 提取位置 ----
#         zombie_positions = [z[1:3] for z in selected_zombies if not np.allclose(z, 0)]
#
#         # ---- 距离统计特征 ----
#         dist_feats = pairwise_distances(zombie_positions)
#
#         # ---- 聚集中心方向与距离 ----
#         if zombie_positions:
#             zombie_positions = np.array(zombie_positions)
#             center = np.mean(zombie_positions, axis=0)  # [rel_x, rel_y]
#         else:
#             center = np.zeros(2)
#
#         # ---- 拼接 ----
#         selected_rows = list(archers) + list(arrows) + selected_zombies
#         flat_obs = np.concatenate(selected_rows + [dist_feats, center])
#         return flat_obs.astype(np.float32)


class CustomPredictFunction:
    """Shared policy version of the predict function."""

    def __init__(self, env):
        package_directory = Path(__file__).resolve().parent
        best_checkpoint = (
            package_directory / "results_multi" / "learner_group" / "learner" / "rl_module"
        ).resolve()

        if not best_checkpoint.exists():
            raise FileNotFoundError(
                f"Checkpoint directory not found: {best_checkpoint}"
            )

        # Only load the shared policy
        self.module = MultiRLModule.from_checkpoint(best_checkpoint)["shared_policy"]

    def __call__(self, observation, agent, *args, **kwargs):
        fwd_ins = {"obs": torch.Tensor(observation).unsqueeze(0)}
        fwd_outputs = self.module.forward_inference(fwd_ins)
        action_dist_class = self.module.get_inference_action_dist_cls()
        action_dist = action_dist_class.from_logits(fwd_outputs["action_dist_inputs"])
        action = action_dist.sample()[0].numpy()
        return action