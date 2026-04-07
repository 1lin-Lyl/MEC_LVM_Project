import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class MADDPGActor(nn.Module):
    """分布式 Actor (每个 UE 独立拥有或参数共享)"""

    def __init__(self, obs_dim, action_dim):
        super(MADDPGActor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()  # 动作输出在 [-1, 1] 之间
        )

    def forward(self, obs):
        return self.net(obs)


class CentralizedCritic(nn.Module):
    """
    集中式 Critic (CTDE 核心)
    能够观测到全局环境（所有 UE 的状态和动作），从而克服非稳态环境问题
    """

    def __init__(self, num_agents, obs_dim, action_dim):
        super(CentralizedCritic, self).__init__()
        # 输入维度: 所有 Agent 的 Obs 和 Action 拼接
        input_dim = num_agents * (obs_dim + action_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # 评估全局/个体的 Q 值
        )

    def forward(self, global_obs, global_actions):
        x = torch.cat([global_obs, global_actions], dim=-1)
        return self.net(x)


class MultiAgentSystem:
    """基于 MADDPG 的多智能体联合调度系统骨架"""

    def __init__(self, num_agents, obs_dim, action_dim):
        self.num_agents = num_agents
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 实例化 Actors 和 集中式 Critic
        self.actors = [MADDPGActor(obs_dim, action_dim).to(self.device) for _ in range(num_agents)]
        self.critic = CentralizedCritic(num_agents, obs_dim, action_dim).to(self.device)

        self.actor_opts = [optim.Adam(a.parameters(), lr=1e-3) for a in self.actors]
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=1e-3)

    def select_actions(self, obs_dict, explore=True):
        """分布式执行 (Decentralized Execution): 每个 Actor 只看自己的局部观测"""
        actions = {}
        for i in range(self.num_agents):
            agent_id = f"ue_{i}"
            obs_tensor = torch.FloatTensor(obs_dict[agent_id]).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.actors[i](obs_tensor).squeeze(0).cpu().numpy()
                if explore:
                    # 引入探索噪声
                    noise = np.random.normal(0, 0.1, size=action.shape)
                    action = np.clip(action + noise, -1.0, 1.0)
            actions[agent_id] = action
        return actions

    def train_step(self, global_obs, global_actions, rewards):
        """
        集中式训练 (Centralized Training) 骨架
        注：实际学术代码中需引入 Replay Buffer 与 Target Network，此处展示梯度传播逻辑
        """
        # 1. 更新 Centralized Critic
        # 计算全局 Q 值，并通过 MSE 损失逼近真实 reward
        # (伪代码省略 Target Q 的贝尔曼方程展开)
        # critic_loss = mse_loss(self.critic(global_obs, global_actions), target_q)

        # 2. 更新 Decentralized Actors
        # 根据 DPG 定理，Actor 的梯度由 Critic 的链式法则引导
        # actor_loss = -self.critic(global_obs, new_global_actions).mean()
        pass