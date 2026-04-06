import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(MLP, self).__init__()
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
    """基于条件扩散模型的 Actor 策略网络"""

    def __init__(self, state_dim, action_dim, n_timesteps=5):
        super(DiffusionActor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_timesteps = n_timesteps
        self.model = MLP(state_dim, action_dim)

    def sample_action(self, state):
        # 强制转换为 int 修复 Bug
        b_size = int(state.size(0))
        device = state.device

        # 传递明确的 Tuple 大小避免 TypeError
        x = torch.randn((b_size, self.action_dim), device=device)

        for t in reversed(range(self.n_timesteps)):
            t_tensor = torch.full((b_size,), t, device=device, dtype=torch.float32)
            noise_pred = self.model(state, x, t_tensor)
            x = x - noise_pred * 0.1

        return torch.tanh(x)