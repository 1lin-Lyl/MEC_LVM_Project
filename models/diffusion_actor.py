import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def __init__(self, state_dim, action_dim, n_timesteps=10):
        super(DiffusionActor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_timesteps = n_timesteps
        self.model = MLP(state_dim, action_dim)

    def sample_action(self, state, action_mask=None):
        """
        采样动作
        :param state: 智能体的观测状态张量 (b_size, state_dim)
        :param action_mask: 合法动作掩码 (b_size, M+1)
        :return: 经过 Gumbel-Softmax 处理的拼接 Action
        """
        # 强制转换为 int 修复 Bug
        b_size = int(state.size(0))
        device = state.device

        # 3.【网络动态规模调整】
        # 当输入 UE 数量 > 20（如 Env B/C）时，增加去噪步数以增强高维复杂空间的搜索精度
        current_timesteps = 30 if b_size > 20 else self.n_timesteps

        # 传递明确的 Tuple 大小避免 TypeError
        x = torch.randn((b_size, self.action_dim), device=device)

        # 逆向去噪过程
        for t in reversed(range(current_timesteps)):
            t_tensor = torch.full((b_size,), t, device=device, dtype=torch.float32)
            noise_pred = self.model(state, x, t_tensor)
            x = x - noise_pred * 0.1

        # 在输出前，分离离散动作 Logits 和 连续的推理步数
        # 动作空间维度: 前 M+1 维是 ES Selection (Logits)，最后 1 维是 Inference Steps
        es_logits = x[:, :-1]
        step_val = x[:, -1:]

        # 1.【融合 Action Mask】
        if action_mask is not None:
            # action_mask 为 1 (合法), 为 0 (非法)。
            # 将非法动作对应的 logit 减去一个极大的数 (-1e9)，强制在后续 Softmax 中被掩蔽
            es_logits = es_logits + (1.0 - action_mask) * (-1e9)

        # 2.【离散化平滑】使用 Gumbel-Softmax 替代 argmax()
        # hard=True 使用直通估计器 (Straight-Through Estimator)，
        # 前向传播输出严格的 One-Hot 向量，反向传播按 Softmax 的平滑梯度更新，保证可导
        es_selection = F.gumbel_softmax(es_logits, tau=1.0, hard=True)

        # 最后 1 维推理步数仍然保持连续映射，使用 tanh 压缩至 [-1, 1]
        step_continuous = torch.tanh(step_val)

        # 拼接重组为最终的 Action
        final_action = torch.cat([es_selection, step_continuous], dim=-1)

        return final_action