import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# ==========================================
# 1. 核心网络架构
# ==========================================
class DiffusionMLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
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
    def __init__(self, state_dim, action_dim, n_timesteps=5):
        super().__init__()
        self.action_dim = action_dim
        self.n_timesteps = n_timesteps
        self.model = DiffusionMLP(state_dim, action_dim)

    def sample_action(self, state):
        b_size = state.size(0)
        device = state.device
        x = torch.randn((b_size, self.action_dim), device=device)

        for t in reversed(range(self.n_timesteps)):
            t_tensor = torch.full((b_size,), float(t), device=device, dtype=torch.float32)
            noise_pred = self.model(state, x, t_tensor)
            x = x - noise_pred * 0.1

        return torch.tanh(x)  # 确保输出严格落在 [-1, 1] 区间


class CentralizedCritic(nn.Module):
    def __init__(self, num_agents, obs_dim, action_dim, hidden_dim=512):
        super().__init__()
        input_dim = num_agents * (obs_dim + action_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, global_obs, global_acts):
        x = torch.cat([global_obs, global_acts], dim=-1)
        return self.net(x)


# ==========================================
# 2. 算法核心控制系统
# ==========================================
class MADiffusionRLSystem:
    def __init__(self, num_agents_train, obs_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = DiffusionActor(obs_dim, action_dim).to(self.device)
        self.critic = CentralizedCritic(num_agents_train, obs_dim, action_dim).to(self.device)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=1e-3)

    def select_actions(self, obs_dict, explore=True):
        agent_ids = list(obs_dict.keys())
        obs_tensor = torch.FloatTensor(np.array([obs_dict[aid] for aid in agent_ids])).to(self.device)

        with torch.no_grad():
            actions = self.actor.sample_action(obs_tensor)
            if explore:
                noise = torch.randn_like(actions) * 0.1
                actions = torch.clamp(actions + noise, -1.0, 1.0)

        actions_np = actions.cpu().numpy()
        return {aid: actions_np[i] for i, aid in enumerate(agent_ids)}

    def train_step(self, obs_dict, action_dict, rewards_dict):
        agent_ids = list(obs_dict.keys())
        num_agents = len(agent_ids)

        obs_tensor = torch.FloatTensor(np.array([obs_dict[aid] for aid in agent_ids])).to(self.device)
        act_tensor = torch.FloatTensor(np.array([action_dict[aid] for aid in agent_ids])).to(self.device)

        g_obs = obs_tensor.view(1, -1)
        g_acts = act_tensor.view(1, -1)

        # 奖励量级压缩，匹配网络最佳敏感度区间
        avg_reward_per_agent = sum(rewards_dict.values()) / num_agents
        scaled_reward = avg_reward_per_agent / 5.0

        # 彻底解决 Broadcasting Bug：确保强制二维对齐 [1, 1]
        g_reward = torch.FloatTensor([[scaled_reward]]).view(-1, 1).to(self.device)

        # --- 1. 更新 Central Critic ---
        q_val = self.critic(g_obs, g_acts).view(-1, 1)
        critic_loss = nn.functional.mse_loss(q_val, g_reward)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_opt.step()

        # --- 2. 更新 Diffusion Actor ---
        new_acts = self.actor.sample_action(obs_tensor)
        new_g_acts = new_acts.view(1, -1)

        actor_loss = -self.critic(g_obs, new_g_acts).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_opt.step()