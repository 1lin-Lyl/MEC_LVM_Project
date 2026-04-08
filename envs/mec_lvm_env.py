import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MultiAgentMECLVMEnvMulti(gym.Env):
    """
    环境: 面向视觉大模型(LVM)的多智能体MEC卸载与推理步数联合优化环境。
    支持三种规模的零样本(Zero-shot)扩展性测试。
    """

    def __init__(self, env_type="B"):
        super().__init__()

        # 支持 Scalability 测试的环境规模配置
        if env_type == "A":  # 小型网络
            self.num_ess = 2
            self.num_ues = 10
        elif env_type == "B":  # 中型主训练网络
            self.num_ess = 5
            self.num_ues = 25
        elif env_type == "C":  # 大型泛化网络
            self.num_ess = 10
            self.num_ues = 50
        else:
            raise ValueError("Unknown env_type")

        # 物理常量定义
        self.B_es = 20.0  # 每台边缘服务器的独立接入下行带宽 (MHz)
        self.P_tx = 0.1  # UE发送功率 (W)
        self.N_0 = 1e-9  # 噪声功率谱密度
        self.eta = 0.5  # LVM 计算复杂度系数 (G Cycles / Mbits)

        self.obs_dim = 3
        self.action_dim = 2  # Action: [ES Selection, Inference Steps]

        self.max_steps = 10  # 每个 Episode 的时隙数量 (为加快训练速度与重组率设置)
        self.current_step = 0
        self.F_ess = None
        self.states = {}

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0

        # 每台 ES 的可用算力随机初始化 (G Cycles / s)
        self.F_ess = [np.random.uniform(150, 250) for _ in range(self.num_ess)]

        self._generate_states()
        return self._get_normalized_states(), {}

    def _generate_states(self):
        """生成每个 UE 的实时任务特征和信道状态"""
        for i in range(self.num_ues):
            self.states[f"ue_{i}"] = np.array([
                np.random.uniform(5, 20),  # S_n: 图像/任务大小 (Mbits)
                np.random.uniform(5, 15),  # F_loc: 本地算力 (G cycles/s)
                np.random.uniform(1e-6, 5e-6)  # h_n: 信道增益
            ], dtype=np.float32)

    def _get_normalized_states(self):
        norm_states = {}
        for k, v in self.states.items():
            norm_v = np.copy(v)
            norm_v[0] /= 20.0  # S_n 归一化
            norm_v[1] /= 15.0  # F_loc 归一化
            norm_v[2] /= 5e-6  # h_n 归一化
            norm_states[k] = norm_v
        return norm_states

    def _calculate_md_vqm(self, steps):
        """多维视频质量模型 (MD-VQM) 的质量效用评分"""
        q_max = 20.0  # 最高分上限
        theta = 0.05  # 指数增长控制参数
        d_min = 5  # 最小生成所需步数
        if steps <= d_min:
            return 0.0
        return q_max * (1 - np.exp(-theta * (steps - d_min)))

    def step(self, action_dict):
        self.current_step += 1

        offload_decisions = {}
        inference_steps = {}
        es_load_count = {k: 0 for k in range(1, self.num_ess + 1)}

        # 1. 动作映射与竞争登记
        for i in range(self.num_ues):
            agent_id = f"ue_{i}"
            act = action_dict[agent_id]

            # Action 0: 卸载决策映射 -> [-1, 1] 映射到 [0, M] (0 代表本地)
            mapped_act0 = (np.clip(act[0], -1.0, 1.0) + 1.0) / 2.0
            target = int(np.round(mapped_act0 * self.num_ess))
            target = np.clip(target, 0, self.num_ess)
            offload_decisions[agent_id] = target

            if target > 0:
                es_load_count[target] += 1

            # Action 1: LVM 推理步数映射 -> [-1, 1] 映射到 [10, 50] 步
            mapped_act1 = (np.clip(act[1], -1.0, 1.0) + 1.0) / 2.0
            steps = int(np.round(mapped_act1 * 40 + 10))
            steps = np.clip(steps, 10, 50)
            inference_steps[agent_id] = steps

        rewards = {}
        infos = {}

        # 2. 物理资源分配结算
        for i in range(self.num_ues):
            agent_id = f"ue_{i}"
            data_size, local_cpu, channel_gain = self.states[agent_id]

            target_es = offload_decisions[agent_id]
            steps = inference_steps[agent_id]

            req_cycles = self.eta * data_size * steps
            vqm_utility = self._calculate_md_vqm(steps)

            delay = 0.0
            energy = 0.0

            if target_es == 0:
                # 本地推理结算
                delay = req_cycles / local_cpu
                energy = 0.5 * data_size * (local_cpu ** 2) * (steps / 100.0)
            else:
                # 边缘计算结算（考虑多 UE 比例公平竞争资源）
                competitors = es_load_count[target_es]
                alloc_bandwidth = self.B_es / competitors
                snr = (self.P_tx * channel_gain) / (alloc_bandwidth * self.N_0)
                trans_rate = alloc_bandwidth * np.log2(1 + snr)

                trans_delay = data_size / trans_rate
                trans_energy = self.P_tx * trans_delay

                alloc_cpu = self.F_ess[target_es - 1] / competitors
                comp_delay = req_cycles / alloc_cpu

                delay = trans_delay + comp_delay
                energy = trans_energy

            # 综合目标函数: 最大化 VQM 质量，惩罚延迟和能耗
            cost = 1.2 * delay + 0.5 * energy
            reward = vqm_utility - cost

            rewards[agent_id] = reward
            infos[agent_id] = {
                'target': target_es, 'steps': steps,
                'vqm': vqm_utility, 'delay': delay, 'energy': energy
            }

        # 刷新下一个时隙的状态
        self._generate_states()

        done = self.current_step >= self.max_steps
        dones = {f"ue_{i}": done for i in range(self.num_ues)}

        return self._get_normalized_states(), rewards, dones, False, infos