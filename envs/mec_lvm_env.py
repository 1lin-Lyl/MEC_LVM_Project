import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MultiAgentMECLVMEnv(gym.Env):
    """
    V2升级版：多智能体、多边缘节点、基于真实扩散模型(Diffusion)物理特征的MEC联合优化环境
    """

    def __init__(self, num_ues=3, num_ess=2):
        super(MultiAgentMECLVMEnv, self).__init__()

        self.num_ues = num_ues  # 用户(Agent)数量
        self.num_ess = num_ess  # 边缘服务器数量

        # 物理常量定义
        self.B_total = 100.0  # 每个ES的总基站带宽 (MHz)
        self.P_tx = 0.1  # UE发送功率 (W)
        self.N_0 = 1e-9  # 噪声功率
        self.eta = 0.5  # LVM 复杂度系数 (Gigacycles / MB / step)

        # 状态空间: [任务数据量, 本地CPU, 信道增益(均值)]
        self.obs_dim = 3
        # 动作空间: [卸载目标 (-1~1映射到0~K), 推理步数 (-1~1映射到10~100)]
        self.action_dim = 2

        # 多智能体空间定义
        self.action_space = spaces.Dict({
            f"ue_{i}": spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
            for i in range(self.num_ues)
        })
        self.observation_space = spaces.Dict({
            f"ue_{i}": spaces.Box(low=0, high=1000, shape=(self.obs_dim,), dtype=np.float32)
            for i in range(self.num_ues)
        })

        self.max_steps = 50
        self.current_step = 0

    def reset(self, seed=None):
        self.current_step = 0
        # ES 计算能力 (GHz)
        self.F_ess = [np.random.uniform(150, 250) for _ in range(self.num_ess)]

        self.states = {}
        for i in range(self.num_ues):
            self.states[f"ue_{i}"] = np.array([
                np.random.uniform(5, 20),  # S_n: 提示词/参考图数据量 (MB)
                np.random.uniform(5, 15),  # F_loc: 本地计算能力 (GHz)
                np.random.uniform(1e-6, 5e-6)  # h_n: 信道增益
            ], dtype=np.float32)

        return self._get_normalized_states(), {}

    def _get_normalized_states(self):
        norm_states = {}
        for k, v in self.states.items():
            norm_v = np.copy(v)
            norm_v[0] /= 20.0  # Max Data Size
            norm_v[1] /= 15.0  # Max Local CPU
            norm_v[2] /= 5e-6  # Max Channel Gain
            norm_states[k] = norm_v
        return norm_states

    def _calculate_md_vqm(self, steps):
        """
        MD-VQM (Multi-Dimensional Video Quality Metric)
        基于 Modified Exponential Function，体现扩散模型边际收益递减
        """
        q_max = 100.0
        theta = 0.05
        d_min = 5
        if steps <= d_min:
            return 0.0
        return q_max * (1 - np.exp(-theta * (steps - d_min)))

    def step(self, action_dict):
        self.current_step += 1

        # 1. 动作解析与资源竞争统计
        offload_decisions = {}
        inference_steps = {}
        es_load_count = {k: 0 for k in range(1, self.num_ess + 1)}  # k=0 是本地

        for i in range(self.num_ues):
            agent_id = f"ue_{i}"
            act = action_dict[agent_id]

            # 映射卸载目标: [-1, 1] -> [0, 1, 2... K]
            target = int((np.clip(act[0], -1.0, 1.0) + 1.0) / 2.0 * self.num_ess)
            target = min(target, self.num_ess)
            offload_decisions[agent_id] = target

            if target > 0:
                es_load_count[target] += 1

            # 映射推理步数: [-1, 1] -> [10, 100]
            steps = int(45 * np.clip(act[1], -1.0, 1.0) + 55)
            inference_steps[agent_id] = steps

        # 2. 物理开销计算 (考虑资源抢占)
        rewards = {}
        infos = {}

        for i in range(self.num_ues):
            agent_id = f"ue_{i}"
            state = self.states[agent_id]
            data_size, local_cpu, channel_gain = state

            target_es = offload_decisions[agent_id]
            steps = inference_steps[agent_id]

            # 任务所需总运算周期 (Gigacycles)
            req_cycles = self.eta * data_size * steps

            # 计算 VQM (视频质量效用)
            vqm_utility = self._calculate_md_vqm(steps)

            delay = 0.0
            energy = 0.0

            if target_es == 0:
                # 本地执行
                delay = req_cycles / local_cpu
                energy = 0.5 * data_size * (local_cpu ** 2) * (steps / 100.0)
            else:
                # 边缘卸载 (发生通信和算力资源竞争)
                competitors = es_load_count[target_es]

                # 香农公式计算传输速率 (均分带宽)
                alloc_bandwidth = self.B_total / competitors
                snr = (self.P_tx * channel_gain) / (alloc_bandwidth * self.N_0)
                trans_rate = alloc_bandwidth * np.log2(1 + snr)

                trans_delay = data_size / trans_rate
                trans_energy = self.P_tx * trans_delay

                # 边缘服务器计算延迟 (均分算力)
                alloc_cpu = self.F_ess[target_es - 1] / competitors
                comp_delay = req_cycles / alloc_cpu

                delay = trans_delay + comp_delay
                energy = trans_energy  # 边缘能耗通常不计入UE端开销

            # 联合效用函数 (最大化 QoE)
            # Reward = VQM - (w1 * Delay + w2 * Energy)
            cost = 1.2 * delay + 0.5 * energy
            reward = vqm_utility - cost

            rewards[agent_id] = reward
            infos[agent_id] = {'target': target_es, 'steps': steps, 'vqm': vqm_utility, 'delay': delay}

        # 3. 状态转移
        self.reset()  # 简化动态：每步刷新独立同分布的信道
        done = self.current_step >= self.max_steps
        dones = {f"ue_{i}": done for i in range(self.num_ues)}

        return self._get_normalized_states(), rewards, dones, False, infos