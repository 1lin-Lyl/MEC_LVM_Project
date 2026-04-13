import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 导入刚刚重构好的扩散模型 Actor
from models.diffusion_actor import DiffusionActor


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
# 算法核心控制系统 (支持 Replay Buffer 与 专家引导)
# ==========================================
class MADiffusionRLSystem:
    def __init__(self, num_agents_train, obs_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = DiffusionActor(obs_dim, action_dim).to(self.device)
        self.critic = CentralizedCritic(num_agents_train, obs_dim, action_dim).to(self.device)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=1e-3)

        # 引入离线强化学习必需的 Replay Buffer
        self.buffer = []
        self.max_buffer_size = 10000
        self.batch_size = 64

    def update_lr(self, lr_actor, lr_critic):
        for param_group in self.actor_opt.param_groups:
            param_group['lr'] = lr_actor
        for param_group in self.critic_opt.param_groups:
            param_group['lr'] = lr_critic

    def select_actions(self, obs_dict, explore=True, noise_scale=0.1, action_mask_dict=None):
        agent_ids = list(obs_dict.keys())
        obs_tensor = torch.FloatTensor(np.array([obs_dict[aid] for aid in agent_ids])).to(self.device)

        mask_tensor = None
        if action_mask_dict is not None:
            mask_tensor = torch.FloatTensor(np.array([action_mask_dict[aid] for aid in agent_ids])).to(self.device)

        with torch.no_grad():
            actions = self.actor.sample_action(obs_tensor, action_mask=mask_tensor)
            if explore:
                noise = torch.randn_like(actions) * noise_scale
                actions = torch.clamp(actions + noise, -1.0, 1.0)

        actions_np = actions.cpu().numpy()
        return {aid: actions_np[i] for i, aid in enumerate(agent_ids)}

    def store_transition(self, obs_dict, action_dict, rewards_dict, next_obs_dict):
        """将专家和模型的探索均存入 Buffer，供离线批处理学习"""
        agent_ids = list(obs_dict.keys())
        num_agents = len(agent_ids)

        obs_arr = np.array([obs_dict[aid] for aid in agent_ids])
        act_arr = np.array([action_dict[aid] for aid in agent_ids])
        next_obs_arr = np.array([next_obs_dict[aid] for aid in agent_ids])

        avg_reward_per_agent = sum(rewards_dict.values()) / num_agents
        scaled_reward = avg_reward_per_agent / 5.0

        self.buffer.append({
            'obs': obs_arr,
            'acts': act_arr,
            'reward': scaled_reward,
            'next_obs': next_obs_arr
        })

        if len(self.buffer) > self.max_buffer_size:
            self.buffer.pop(0)

    def train_step(self):
        """基于 Replay Buffer 随机采样进行批处理学习"""
        if len(self.buffer) < self.batch_size:
            return

        # 随机抽取 Batch
        batch_indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer[i] for i in batch_indices]

        obs_batch = torch.FloatTensor(np.array([b['obs'] for b in batch])).to(self.device)  # (B, N, obs_dim)
        act_batch = torch.FloatTensor(np.array([b['acts'] for b in batch])).to(self.device)  # (B, N, act_dim)
        reward_batch = torch.FloatTensor(np.array([[b['reward']] for b in batch])).to(self.device)  # (B, 1)

        B, N, obs_dim = obs_batch.shape
        act_dim = act_batch.shape[-1]

        g_obs = obs_batch.view(B, -1)
        g_acts = act_batch.view(B, -1)

        # --- 1. 更新 Central Critic ---
        q_val = self.critic(g_obs, g_acts).view(-1, 1)
        critic_loss = nn.functional.mse_loss(q_val, reward_batch)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_opt.step()

        # --- 2. 更新 Diffusion Actor ---
        # 展平所有智能体维度供 Actor 处理，反传时依然组合为全局视角骗过 Critic
        flat_obs = obs_batch.view(B * N, obs_dim)
        new_acts = self.actor.sample_action(flat_obs)
        new_g_acts = new_acts.view(B, -1)

        actor_loss = -self.critic(g_obs, new_g_acts).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_opt.step()

    def reset_buffer(self):
        # 离线强化学习算法保留 Buffer，不应清空
        pass