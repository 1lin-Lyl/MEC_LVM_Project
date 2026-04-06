import numpy as np
import gym
from gym import spaces


class MECLVMEnv(gym.Env):
    """
    面向移动边缘计算(MEC)场景的视觉大模型(LVM)推理与任务卸载环境。
    """

    def __init__(self):
        super(MECLVMEnv, self).__init__()

        # 状态空间维度 = 4
        # [0]: 任务数据量 Data Size (MB)
        # [1]: 边缘服务器计算能力 Edge CPU (GHz)
        # [2]: 本地设备计算能力 Local CPU (GHz)
        # [3]: 无线网络带宽 Bandwidth (Mbps)
        self.state_dim = 4
        self.action_dim = 2

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=150, shape=(self.state_dim,), dtype=np.float32)

        self.max_steps = 50
        self.current_step = 0
        self.state = None

    def _get_normalized_state(self, state):
        """
        [学术优化] 状态归一化 (State Normalization)
        由于数据量、CPU、带宽的量纲和绝对值差异很大，直接输入神经网络容易导致梯度不稳定。
        将状态除以各自的最大近似值，使其大致分布在 [0, 1] 之间，这是深度学习的标准操作。
        """
        norm_state = np.zeros_like(state)
        norm_state[0] = state[0] / 50.0  # 数据量最大约 50
        norm_state[1] = state[1] / 120.0  # 边缘 CPU 最大约 120
        norm_state[2] = state[2] / 15.0  # 本地 CPU 最大约 15
        norm_state[3] = state[3] / 100.0  # 带宽最大约 100
        return norm_state

    def reset(self):
        self.current_step = 0
        raw_state = np.array([
            np.random.uniform(10, 50),
            np.random.uniform(80, 120),
            np.random.uniform(5, 15),
            np.random.uniform(10, 100)
        ], dtype=np.float32)
        self.state = raw_state
        return self._get_normalized_state(self.state)

    def step(self, action):
        self.current_step += 1

        offload_ratio = np.clip((action[0] + 1) / 2.0, 0.0, 1.0)
        compress_ratio = np.clip((action[1] + 1) / 2.0, 0.0, 1.0)

        data_size, edge_cpu, local_cpu, bandwidth = self.state

        # 计算开销
        local_data = data_size * (1 - offload_ratio)
        local_delay = local_data / local_cpu
        local_energy = 0.5 * local_data * (local_cpu ** 2)

        edge_data = data_size * offload_ratio * compress_ratio
        trans_delay = edge_data / bandwidth
        edge_delay = edge_data / edge_cpu
        trans_energy = 0.2 * edge_data

        total_delay = max(local_delay, trans_delay + edge_delay)
        total_energy = local_energy + trans_energy

        # 精度损失
        accuracy_penalty = 10 * (1 - compress_ratio) ** 2

        # 奖励函数
        cost = (1.0 * total_delay) + (0.1 * total_energy) + (0.5 * accuracy_penalty)
        reward = -cost

        self.state = np.array([
            np.random.uniform(10, 50),
            np.random.uniform(80, 120),
            np.random.uniform(5, 15),
            np.random.uniform(10, 100)
        ], dtype=np.float32)

        done = self.current_step >= self.max_steps
        info = {'delay': total_delay, 'energy': total_energy, 'penalty': accuracy_penalty}

        return self._get_normalized_state(self.state), float(reward), done, info