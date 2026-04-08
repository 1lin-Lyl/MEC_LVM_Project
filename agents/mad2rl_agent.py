import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# ==========================================
# 1. 网络架构定义
# ==========================================
class DiffusionMLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        # 接收 State, 污染 Action(x_t), 和时间步 t
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state, action, t):
        t = t.unsqueeze(-1) if t.dim() == 1 else t
        x = torch.cat([state, action, t], dim=-1)
        return self.net(x)


class DiffusionActor(nn.Module):
    """基于条件扩散模型的 Actor (支持批量前向推导且可微分)"""

    def __init__(self, state_dim, action_dim, n_timesteps=5):
        super().__init__()
        self.action_dim = action_dim
        self.n_timesteps = n_timesteps
        self.model = DiffusionMLP(state_dim, action_dim)

    def sample_action(self, state):
        b_size = state.size(0)
        device = state.device
        # 纯随机高斯噪声作为初始态
        x = torch.randn((b_size, self.action_dim), device=device)

        # 逐步去噪 (反向扩散)，全过程保持 PyTorch 计算图以便 DPG 梯度反传
        for t in reversed(range(self.n_timesteps)):
            t_tensor = torch.full((b_size,), float(t), device=device, dtype=torch.float32)
            noise_pred = self.model(state, x, t_tensor)
            x = x - noise_pred * 0.1

        return torch.tanh(x)


class CentralizedCritic(nn.Module):
    def __init__(self, num_agents, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        input_dim = num_agents * (obs_dim + action_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, global_obs, global_acts):
        x = torch.cat([global_obs, global_acts], dim=-1)
        return self.net(x)


# ==========================================
# 2. 算法核心控制系统
# ==========================================
class MADiffusionRLSystem:
    """ Parameter-Sharing MA-Diffusion-RL """

    def __init__(self, num_agents_train, obs_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 【核心创新】全部UE共享同一个去中心化Actor，不仅训练快，更无缝支持泛化规模测试！
        self.actor = DiffusionActor(obs_dim, action_dim).to(self.device)

        # 中心化 Critic 仅在主训练环节被使用，以全局视角辅助收敛
        self.critic = CentralizedCritic(num_agents_train, obs_dim, action_dim).to(self.device)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=3e-4)

    def select_actions(self, obs_dict, explore=True):
        agent_ids = list(obs_dict.keys())
        # 批量并行推理
        obs_tensor = torch.FloatTensor(np.array([obs_dict[aid] for aid in agent_ids])).to(self.device)

        with torch.no_grad():
            actions = self.actor.sample_action(obs_tensor)
            if explore:
                noise = torch.randn_like(actions) * 0.1
                actions = torch.clamp(actions + noise, -1.0, 1.0)

        actions_np = actions.cpu().numpy()
        return {aid: actions_np[i] for i, aid in enumerate(agent_ids)}

    def train_step(self, obs_dict, action_dict, rewards_dict):
        """Deterministic Policy Gradient (DPG) 更新 Diffusion Actor 和 Critic"""
        agent_ids = list(obs_dict.keys())
        obs_tensor = torch.FloatTensor(np.array([obs_dict[aid] for aid in agent_ids])).to(self.device)
        act_tensor = torch.FloatTensor(np.array([action_dict[aid] for aid in agent_ids])).to(self.device)

        # 压平为全局态
        g_obs = obs_tensor.view(1, -1)
        g_acts = act_tensor.view(1, -1)

        # 系统总奖励反馈
        system_reward = sum(rewards_dict.values()) / 10.0  # Reward Scaling
        g_reward = torch.FloatTensor([system_reward]).to(self.device)

        # --- 1. 更新 Central Critic ---
        q_val = self.critic(g_obs, g_acts).squeeze()
        critic_loss = nn.functional.mse_loss(q_val, g_reward)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.critic_opt.step()

        # --- 2. 更新 Diffusion Actor ---
        # 允许梯度直接穿过 Actor 去噪链进行反向传播
        new_acts = self.actor.sample_action(obs_tensor)
        new_g_acts = new_acts.view(1, -1)

        # 目标是最大化 Q 值，即最小化 -Q
        actor_loss = -self.critic(g_obs, new_g_acts).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.actor_opt.step()