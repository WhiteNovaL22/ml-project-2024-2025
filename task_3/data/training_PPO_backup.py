#!/usr/bin/env python3
# encoding: utf-8
"""
Training a single agent using Ray RLLib.
Method: PPO.
"""

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

# 环境包装器，此处进行特征工程
class CustomWrapper(BaseWrapper):
    # This is an example of a custom wrapper that flattens the symbolic vector state of the environment
    # Wrapper are useful to inject state pre-processing or feature that does not need to be learned by the agent

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return  spaces.flatten_space(super().observation_space(agent))

    def observe(self, agent: AgentID) -> ObsType | None:
        obs = super().observe(agent)
        flat_obs = obs.flatten()
        return flat_obs


class CustomPredictFunction(Callable):
    """ This is an example of an instantiation of the CustomPredictFunction that loads a trained RLLib algorithm from
    a checkpoint and extract the policies from it"""

    def __init__(self, env):

        # Here you should load your trained model(s) from a checkpoint in your folder
        best_checkpoint = (Path("results") / "learner_group" / "learner" / "rl_module").resolve()
        self.modules = MultiRLModule.from_checkpoint(best_checkpoint)

    def __call__(self, observation, agent, *args, **kwargs):
        rl_module = self.modules[agent]
        fwd_ins = {"obs": torch.Tensor(observation).unsqueeze(0)}
        fwd_outputs = rl_module.forward_inference(fwd_ins)
        action_dist_class = rl_module.get_inference_action_dist_cls()
        action_dist = action_dist_class.from_logits(
            fwd_outputs["action_dist_inputs"]
        )
        action = action_dist.sample()[0].numpy()
        return action



# 算法配置：环境名称，所有 agent 对应的策略名集合（一般等于 agent 的 ID），希望训练的策略（agent ID）
def algo_config(id_env, policies, policies_to_train):


    config = (
        PPOConfig() # 默认PPO配置
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment(env=id_env, disable_env_checking=True)
        .env_runners(num_env_runners=4) # 设置同时运行几个环境实例。可以增加以加快训练
        .multi_agent(
            policies={x for x in policies},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id, # 分配策略：每个 agent 使用自己的同名策略
            policies_to_train=policies_to_train, # 指定哪些策略会被训练；其余的将使用但不更新
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    # 网络结构为两层全连接，每层 64 个神经元。
                    x: RLModuleSpec(module_class=PPOTorchRLModule, model_config={"fcnet_hiddens": [64, 64]})
                    if x in policies_to_train
                    else
                    RLModuleSpec(module_class=RandomRLModule) # 对于未训练的 agent：使用随机策略（占位）
                    for x in policies},
            ))
        .training(
            train_batch_size=1024, # 每轮优化使用多少样本
            lr=1e-4, # 学习率
            gamma=0.99, # 折扣因子，衡量未来奖励的重要性
            # clip_param=0.2,
            use_critic=True,
            use_gae=True,
            lambda_=0.95,
            # grad_clip=0.5,
            # vf_clip_param=10.0,
            # vf_loss_coeff=1.0,
            # entropy_coeff=0.01,
        ) # 此处进行参数调优
        .debugging(log_level="ERROR")

    )

    return config

# 连续20轮无进步则停止
def make_early_stopper(agent_id, patience):
    best_reward = float("-inf")
    no_improvement_counter = 0

    def should_stop(result):
        nonlocal best_reward, no_improvement_counter
        if "env_runners" in result and "agent_episode_returns_mean" in result["env_runners"]:
            rewards = result["env_runners"]["agent_episode_returns_mean"]
            current_reward = rewards.get(agent_id, 0)
            print(f"Current reward for {agent_id}: {current_reward}")

            if current_reward > best_reward:
                best_reward = current_reward
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1

            if no_improvement_counter >= patience:
                print(f"[Early Stop] No improvement for {patience} iterations. Best: {best_reward}")
                return True
        return False

    return should_stop


# 执行训练：包装的环境，模型路径，训练次数
def training(env, checkpoint_path, max_iterations = 500):

    # 把原始环境转换为 RLLib 可用的格式
    # Translating the PettingZoo environment to an RLLib environment.
    # Note: RLLib use a parallelized version of the environment.
    rllib_env = ParallelPettingZooEnv(pettingzoo.utils.conversions.aec_to_parallel(env))
    id_env = "knights_archers_zombies_v10"
    register_env(id_env, lambda config: rllib_env) # 当你调用这个 id_env 时，就返回你自定义的环境对象 rllib_env，忽略外部传参

    # Fix seeds
    np.random.seed(42)
    torch.manual_seed(42)

    # Define the configuration for the PPO algorithm
    policies = [x for x in env.agents] # 返回当前环境中所有的 agent 名称
    policies_to_train = policies # 所有 agent 都设置为训练对象
    config = algo_config(id_env, policies, policies_to_train) # 调用 algo_config() 函数，生成 RLLib 的 PPO 配置对象

    # Train the model
    algo = config.build() #  创建一个可以运行的 PPO 算法实例，准备开始训练
    early_stop = make_early_stopper(agent_id="archer_0", patience=50) # 早停条件
    for i in range(max_iterations):
        result = algo.train() # result 包含了当前这轮训练中得到的各种指标（奖励、损失等）
        result.pop("config") # 删除 config 字段（节省内存、避免打印太冗长）
        if "env_runners" in result and "agent_episode_returns_mean" in result["env_runners"]:
            print(i, result["env_runners"]["agent_episode_returns_mean"]) # 打印这一段提取训练过程中 agent 的平均 episodic reward
            # if early_stop(result):
            if result["env_runners"]["agent_episode_returns_mean"]["archer_0"] > 50:
                break # 如果某个 agent 的平均回报超过阈值（例如 archer_0 > 5），就提前终止训练。
        if (i + 1) % 5 == 0:
            save_result = algo.save(checkpoint_path)
            path_to_checkpoint = save_result.checkpoint.path
            print(
                "An Algorithm checkpoint has been created inside directory: "
                f"'{path_to_checkpoint}'."
            ) # 每 5 轮保存一次模型 checkpoint，控制台会打印出保存路径

if __name__ == "__main__":

    num_agents = 1
    visual_observation = False

    # Create the PettingZoo environment for training
    env = create_environment(num_agents=num_agents, visual_observation=visual_observation)
    env = CustomWrapper(env)

    # Running training routine
    checkpoint_path = str(Path("results").resolve())
    training(env, checkpoint_path, max_iterations = 500)
