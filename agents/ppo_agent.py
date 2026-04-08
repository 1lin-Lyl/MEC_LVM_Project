import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class MAPPOActor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, act_dim), nn.Tanh()
        )
        self.log_std = nn.Parameter(torch.zeros(1, act_dim))

    def forward(self, obs):
        mean = self.net(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        std = torch.clamp(std, min=1e-3, max=10.0)  # 【核心修复】防止 std 过小导致分布计算时产生 NaN
        return mean, std


class MAPPOCritic(nn.Module):
    def __init__(self, num_agents, obs_dim):
        super().__init__()
        input_dim = num_agents * obs_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, global_obs):
        return self.net(global_obs)


class MAPPOAgentSystem:
    def __init__(self, num_agents_train, obs_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = MAPPOActor(obs_dim, action_dim).to(self.device)
        self.critic = MAPPOCritic(num_agents_train, obs_dim).to(self.device)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.buffer = []

    def select_actions(self, obs_dict, explore=True):
        agent_ids = list(obs_dict.keys())
        obs_tensor = torch.FloatTensor(np.array([obs_dict[aid] for aid in agent_ids])).to(self.device)

        with torch.no_grad():
            mean, std = self.actor(obs_tensor)
            if explore:
                dist = torch.distributions.Normal(mean, std)
                actions = dist.sample()
                log_probs = dist.log_prob(actions).sum(dim=-1)
            else:
                actions = mean
                log_probs = torch.zeros(len(agent_ids)).to(self.device)

            actions = torch.clamp(actions, -1.0, 1.0)

        return {aid: actions[i].cpu().numpy() for i, aid in enumerate(agent_ids)}, log_probs

    def store_transition(self, obs_dict, action_dict, log_probs, rewards_dict):
        agent_ids = list(obs_dict.keys())
        obs_arr = np.array([obs_dict[aid] for aid in agent_ids])
        act_arr = np.array([action_dict[aid] for aid in agent_ids])

        # 标尺缩放，对齐 Diffusion
        avg_reward_per_agent = sum(rewards_dict.values()) / len(agent_ids)
        sys_reward = avg_reward_per_agent / 5.0

        # 【核心修复】移除 .sum().item()，必须保留智能体维度的对数概率向量以防止梯度爆炸
        self.buffer.append({
            'obs': obs_arr, 'acts': act_arr,
            'log_probs': log_probs.cpu().numpy(),
            'reward': sys_reward
        })

    def reset_buffer(self):
        self.buffer = []

    def update(self):
        if len(self.buffer) == 0: return

        obs_batch = torch.FloatTensor(np.array([b['obs'] for b in self.buffer])).to(self.device)
        act_batch = torch.FloatTensor(np.array([b['acts'] for b in self.buffer])).to(self.device)
        # 【核心修复】还原为二维张量 [Batch, Agents]
        old_log_probs = torch.FloatTensor(np.array([b['log_probs'] for b in self.buffer])).to(self.device)

        rewards = [b['reward'] for b in self.buffer]
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + 0.95 * R
            returns.insert(0, R)

        returns = torch.FloatTensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        returns = returns.view(-1, 1)

        b_size, n_agents, obs_dim = obs_batch.shape
        g_obs_batch = obs_batch.view(b_size, -1)

        for _ in range(4):
            mean, std = self.actor(obs_batch)
            dist = torch.distributions.Normal(mean, std)
            # 【核心修复】只对特征维度求和，保留各个 Agent 的独立维度
            new_log_probs = dist.log_prob(act_batch).sum(dim=-1)

            values = self.critic(g_obs_batch).view(-1, 1)
            advantages = (returns - values.detach())

            # 此时 new_log_probs 和 old_log_probs 为 [Batch, Agents]
            # advantages 为 [Batch, 1]，通过广播机制(Broadcasting)自动作用到每个智能体的截断损失上
            ratio = torch.exp(new_log_probs - old_log_probs.detach())
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = nn.functional.mse_loss(values, returns)

            self.actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_opt.step()

            self.critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            self.critic_opt.step()

        self.buffer.clear()