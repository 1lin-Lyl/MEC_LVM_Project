import torch
import torch.optim as optim
from models.diffusion_actor import DiffusionActor
from models.critic_network import Critic


class DiffusionRLAgent:
    """基于扩散模型的强化学习算法智能体"""

    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = DiffusionActor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)

        # [学术优化] 增加目标 Critic 网络 (Target Network)
        # 这是防止长时间训练出现 Q 值过估计现象的标准操作
        self.target_critic = Critic(state_dim, action_dim).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=3e-4)

    def select_action(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor.sample_action(state_t).squeeze(0).cpu().numpy()
        return action

    def train(self, state, action, reward, next_state, done):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_t = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        reward_t = torch.FloatTensor([reward]).unsqueeze(0).to(self.device)

        q_val = self.critic(state_t, action_t)
        critic_loss = torch.nn.functional.mse_loss(q_val, reward_t)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        new_action = self.actor.sample_action(state_t)
        actor_loss = -self.critic(state_t, new_action).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # [学术优化] 目标网络的软更新 (Soft Update)
        tau = 0.005
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)