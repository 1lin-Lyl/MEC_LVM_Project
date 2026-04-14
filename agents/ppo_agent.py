import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class MAPPOActor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        # 动作空间分离：前 act_dim-1 维为离散 ES 选择，最后 1 维为连续步数
        self.es_dim = act_dim - 1

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU()
        )

        # 离散动作的 Logit 分支
        self.es_logits_head = nn.Linear(256, self.es_dim)

        # 连续推理步数的 Mean 分支
        self.step_mean_head = nn.Sequential(
            nn.Linear(256, 1), nn.Tanh()
        )
        self.step_log_std = nn.Parameter(torch.zeros(1, 1))

    def forward(self, obs, action_mask=None):
        x = self.net(obs)

        # 1. ES 离散选择分支
        es_logits = self.es_logits_head(x)
        if action_mask is not None:
            # 【核心掩码机制】强制把非法节点的 logit 抹成 -1e9，Softmax 之后概率绝对为 0
            es_logits = es_logits + (1.0 - action_mask) * (-1e9)

        # 2. 步数连续控制分支
        step_mean = self.step_mean_head(x)
        std = torch.exp(self.step_log_std).expand_as(step_mean)
        std = torch.clamp(std, min=1e-3, max=10.0)

        return es_logits, step_mean, std


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

    def update_lr(self, lr_actor, lr_critic):
        for param_group in self.actor_opt.param_groups:
            param_group['lr'] = lr_actor
        for param_group in self.critic_opt.param_groups:
            param_group['lr'] = lr_critic

    def select_actions(self, obs_dict, explore=True, action_mask_dict=None):
        agent_ids = list(obs_dict.keys())
        obs_tensor = torch.FloatTensor(np.array([obs_dict[aid] for aid in agent_ids])).to(self.device)

        mask_tensor = None
        if action_mask_dict is not None:
            mask_tensor = torch.FloatTensor(np.array([action_mask_dict[aid] for aid in agent_ids])).to(self.device)

        with torch.no_grad():
            es_logits, step_mean, step_std = self.actor(obs_tensor, action_mask=mask_tensor)

            # 定义离散的分类分布 (Categorical) 和连续的正态分布 (Normal)
            dist_es = torch.distributions.Categorical(logits=es_logits)
            dist_step = torch.distributions.Normal(step_mean, step_std)

            if explore:
                es_acts = dist_es.sample()
                step_acts = dist_step.sample()
                # 联合动作概率等于独立维度的对数概率求和
                log_probs = dist_es.log_prob(es_acts) + dist_step.log_prob(step_acts).sum(dim=-1)
            else:
                es_acts = torch.argmax(es_logits, dim=-1)
                step_acts = step_mean
                log_probs = torch.zeros(len(agent_ids)).to(self.device)

            step_acts = torch.clamp(step_acts, -1.0, 1.0)

            # 兼容环境 step 解析的连续向量格式：
            # 将选择的类别变成独热编码，再映射到类似连续的 [-1, 1]，环境中的 argmax() 能够无损反解
            es_one_hot = torch.zeros_like(es_logits).scatter_(-1, es_acts.unsqueeze(-1), 1.0)
            es_out = es_one_hot * 2.0 - 1.0

            actions = torch.cat([es_out, step_acts], dim=-1)

        return {aid: actions[i].cpu().numpy() for i, aid in enumerate(agent_ids)}, log_probs

    def store_transition(self, obs_dict, action_dict, log_probs, rewards_dict, action_mask_dict=None):
        agent_ids = list(obs_dict.keys())
        obs_arr = np.array([obs_dict[aid] for aid in agent_ids])
        act_arr = np.array([action_dict[aid] for aid in agent_ids])

        avg_reward_per_agent = sum(rewards_dict.values()) / len(agent_ids)
        sys_reward = avg_reward_per_agent / 5.0

        mask_arr = np.array([action_mask_dict[aid] for aid in agent_ids]) if action_mask_dict is not None else None

        if isinstance(log_probs, torch.Tensor):
            lp_arr = log_probs.cpu().numpy()
        else:
            lp_arr = log_probs

        self.buffer.append({
            'obs': obs_arr, 'acts': act_arr,
            'log_probs': lp_arr,
            'reward': sys_reward,
            'mask': mask_arr
        })

    def reset_buffer(self):
        self.buffer = []

    def update(self, entropy_coef=0.01):
        if len(self.buffer) == 0: return

        obs_batch = torch.FloatTensor(np.array([b['obs'] for b in self.buffer])).to(self.device)
        act_batch = torch.FloatTensor(np.array([b['acts'] for b in self.buffer])).to(self.device)
        old_log_probs = torch.FloatTensor(np.array([b['log_probs'] for b in self.buffer])).to(self.device)

        masks_list = [b['mask'] for b in self.buffer]
        if all(m is not None for m in masks_list):
            mask_batch = torch.FloatTensor(np.array(masks_list)).to(self.device)
        else:
            mask_batch = None

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
            es_logits, step_mean, step_std = self.actor(obs_batch, action_mask=mask_batch)
            dist_es = torch.distributions.Categorical(logits=es_logits)
            dist_step = torch.distributions.Normal(step_mean, step_std)

            # 反解存储进缓冲区的动作
            es_acts = torch.argmax(act_batch[:, :, :-1], dim=-1)
            step_acts = act_batch[:, :, -1:]

            new_log_probs = dist_es.log_prob(es_acts) + dist_step.log_prob(step_acts).sum(dim=-1)

            # PPO 混合熵鼓励探索
            entropy = (dist_es.entropy() + dist_step.entropy().sum(dim=-1)).mean()

            values = self.critic(g_obs_batch).view(-1, 1)
            advantages = (returns - values.detach())

            ratio = torch.exp(new_log_probs - old_log_probs.detach())
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * advantages

            actor_loss = -torch.min(surr1, surr2).mean() - entropy_coef * entropy
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