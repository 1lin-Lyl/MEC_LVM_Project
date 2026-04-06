import torch
import torch.nn as nn
import torch.optim as optim


class PPOActor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        return self.net(state)


class PPOAgent:
    """对比基线1: PPO / PG 智能体"""

    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = PPOActor(state_dim, action_dim).to(self.device)
        # 降低学习率，防止在复杂环境中步伐过大导致崩溃
        self.opt = optim.Adam(self.actor.parameters(), lr=3e-4)

    def select_action(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mean = self.actor(state_t)
            noise = torch.randn_like(mean) * 0.05
            action = torch.clamp(mean + noise, -1.0, 1.0).squeeze(0).cpu().numpy()
        return action

    def train(self, state, action, reward, next_state, done):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_t = torch.FloatTensor(action).unsqueeze(0).to(self.device)

        # [学术优化] 奖励缩放 (Reward Scaling)
        # 防止 PPO 因获取到绝对值几千的负奖励而导致梯度爆炸(Gradient Explosion)。
        # 将其缩小 1000 倍，使得 loss 维持在一个合理的数值范围内。
        scaled_reward = reward / 1000.0

        pred_action = self.actor(state_t)
        loss = torch.nn.functional.mse_loss(pred_action, action_t) * (-scaled_reward)

        self.opt.zero_grad()
        loss.backward()
        # 加入梯度裁剪(Gradient Clipping)，进一步保证 PPO 的稳定性
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.opt.step()