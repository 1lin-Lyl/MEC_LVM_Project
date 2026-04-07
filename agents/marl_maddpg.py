import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class MADDPGActor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(MADDPGActor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )

    def forward(self, obs):
        return self.net(obs)


class CentralizedCritic(nn.Module):
    def __init__(self, num_agents, obs_dim, action_dim):
        super(CentralizedCritic, self).__init__()
        input_dim = num_agents * (obs_dim + action_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, global_obs, global_actions):
        x = torch.cat([global_obs, global_actions], dim=-1)
        return self.net(x)


class MultiAgentSystem:
    """MADDPG (多智能体深度确定性策略梯度) 系统核心实现"""

    def __init__(self, num_agents, obs_dim, action_dim):
        self.num_agents = num_agents
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actors = [MADDPGActor(obs_dim, action_dim).to(self.device) for _ in range(num_agents)]
        self.critic = CentralizedCritic(num_agents, obs_dim, action_dim).to(self.device)

        self.actor_opts = [optim.Adam(a.parameters(), lr=1e-3) for a in self.actors]
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=3e-3)

    def select_actions(self, obs_dict, explore=True):
        actions = {}
        for i in range(self.num_agents):
            agent_id = f"ue_{i}"
            obs_tensor = torch.FloatTensor(obs_dict[agent_id]).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.actors[i](obs_tensor).squeeze(0).cpu().numpy()
                if explore:
                    # 递减的探索噪声
                    noise = np.random.normal(0, 0.1, size=action.shape)
                    action = np.clip(action + noise, -1.0, 1.0)
            actions[agent_id] = action
        return actions

    def train_step(self, global_obs, global_actions, rewards):
        """【补充完整】集中式训练 (Centralized Training) 核心逻辑"""
        # 将字典转换为统一维度的 Tensor
        obs_list = [torch.FloatTensor(global_obs[f"ue_{i}"]).to(self.device) for i in range(self.num_agents)]
        act_list = [torch.FloatTensor(global_actions[f"ue_{i}"]).to(self.device) for i in range(self.num_agents)]

        g_obs = torch.cat(obs_list, dim=-1).unsqueeze(0)
        g_acts = torch.cat(act_list, dim=-1).unsqueeze(0)

        # 使用总体系统回报作为反馈
        system_reward = sum(rewards.values()) / 100.0  # 奖励缩放，防梯度爆炸
        g_reward = torch.FloatTensor([system_reward]).unsqueeze(0).to(self.device)

        # 1. 更新 Centralized Critic
        q_val = self.critic(g_obs, g_acts)
        critic_loss = nn.functional.mse_loss(q_val, g_reward)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # 2. 更新 Decentralized Actors
        new_act_list = []
        for i in range(self.num_agents):
            new_act_list.append(self.actors[i](obs_list[i].unsqueeze(0)))
        new_g_acts = torch.cat(new_act_list, dim=-1)

        # 最大化全局 Q 值 (即最小化 -Q)
        actor_loss = -self.critic(g_obs, new_g_acts).mean()

        for opt in self.actor_opts:
            opt.zero_grad()
        actor_loss.backward()
        for opt in self.actor_opts:
            opt.step()