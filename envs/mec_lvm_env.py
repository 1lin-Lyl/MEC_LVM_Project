import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MultiAgentMECLVMEnvMulti(gym.Env):
    """
    环境: 面向视觉大模型(LVM)的多智能体MEC卸载与推理步数联合优化环境。
    """

    def __init__(self, env_type="B"):
        super().__init__()

        # 真实环境规模配置
        if env_type == "A":  # 小型测试网络
            self.num_ess = 2
            self.num_ues = 10
        elif env_type == "B":  # 中型主训练网络
            self.num_ess = 5
            self.num_ues = 25
        elif env_type == "C":  # 大型扩展网络
            self.num_ess = 10
            self.num_ues = 50
        else:
            raise ValueError("Unknown env_type")

        # 核心物理常量
        self.B_es = 20.0  # 每台边缘服务器的独立接入下行带宽 (MHz)
        self.P_tx = 0.1  # UE发送功率 (W)
        self.N_0 = 1e-9  # 噪声功率谱密度
        self.eta = 0.5  # LVM 计算复杂度系数 (G Cycles / Mbits)
        self.P_es_active = 50.0  # 边缘服务器工作状态推理功率 (W)

        # 状态与动作维度
        self.obs_dim = 3 + 2 * self.num_ess
        self.action_dim = self.num_ess + 2

        self.max_steps = 10
        self.current_step = 0
        self.F_ess = None
        self.states = {}
        self.es_loads = [0] * self.num_ess

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        self.es_loads = [0] * self.num_ess

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
        """融合全局视角，输出严格归一化的观测向量"""
        norm_states = {}
        f_norm = np.array(self.F_ess) / 250.0
        l_norm = np.array(self.es_loads) / float(self.num_ues)

        for k, v in self.states.items():
            norm_v = np.copy(v)
            norm_v[0] /= 20.0
            norm_v[1] /= 15.0
            norm_v[2] /= 5e-6
            norm_states[k] = np.concatenate([norm_v, f_norm, l_norm]).astype(np.float32)
        return norm_states

    def _calculate_md_vqm(self, steps):
        """多维视频质量模型 (MD-VQM)"""
        q_max = 20.0
        theta = 0.05
        d_min = 5
        if steps <= d_min:
            return 0.0
        return q_max * (1 - np.exp(-theta * (steps - d_min)))

    def step(self, action_dict):
        self.current_step += 1

        # 安全防护
        for agent_id, act in action_dict.items():
            if np.isnan(act).any():
                action_dict[agent_id] = np.zeros_like(act)

        offload_decisions = {}
        inference_steps = {}
        es_load_count = {k: 0 for k in range(1, self.num_ess + 1)}

        # 1. 动作映射解构
        for i in range(self.num_ues):
            agent_id = f"ue_{i}"
            act = action_dict[agent_id]

            target = int(np.argmax(act[:self.num_ess + 1]))
            offload_decisions[agent_id] = target
            if target > 0:
                es_load_count[target] += 1

            step_val = act[-1]
            steps = int(np.round((np.clip(step_val, -1.0, 1.0) + 1.0) / 2.0 * 40 + 10))
            inference_steps[agent_id] = np.clip(steps, 10, 50)

        self.es_loads = [es_load_count[m + 1] for m in range(self.num_ess)]

        rewards = {}
        infos = {}

        # 2. 物理资源与【真实能耗】分配结算
        for i in range(self.num_ues):
            agent_id = f"ue_{i}"
            data_size, local_cpu, channel_gain = self.states[agent_id]

            target_es = offload_decisions[agent_id]
            steps = inference_steps[agent_id]

            req_cycles = self.eta * data_size * steps
            vqm_utility = self._calculate_md_vqm(steps)

            delay = 0.0
            energy = 0.0  # 真实物理系统能耗 (J)

            if target_es == 0:
                local_cpu = max(local_cpu, 1e-9)
                delay = req_cycles / local_cpu
                # 【修复】本地动态功率计算：基于CPU频率(近似平方关系/10以匹配手机芯片1-10W功耗)
                p_local = 0.5 * (local_cpu ** 2) / 10.0
                energy = p_local * delay
            else:
                competitors = es_load_count[target_es]
                alloc_bandwidth = self.B_es / competitors
                snr = (self.P_tx * channel_gain) / (alloc_bandwidth * self.N_0)
                trans_rate = alloc_bandwidth * np.log2(1 + snr)

                trans_delay = data_size / trans_rate
                trans_energy = self.P_tx * trans_delay

                alloc_cpu = self.F_ess[target_es - 1] / competitors
                comp_delay = req_cycles / alloc_cpu

                # 【修复】增加边缘服务器推理的庞大能耗 (假设50W功率)
                comp_energy = self.P_es_active * comp_delay

                delay = trans_delay + comp_delay
                energy = trans_energy + comp_energy  # 系统总真实能耗包含传输 + ES计算

            # 【修复】将惩罚和成本的量级隔离开，不污染真实能耗返回
            clip_delay = min(delay, 15.0)
            clip_energy = min(energy, 500.0)  # RL可感知的截断最大能耗，防止极值破坏网络
            penalty = -5.0 if delay >= 15.0 else 0.0

            # 将 RL 的 Cost 严格对齐至 0~20 的量级
            cost = 5.0 * (clip_delay / 15.0) + 2.0 * (clip_energy / 500.0)
            reward = vqm_utility - cost + penalty

            rewards[agent_id] = reward
            infos[agent_id] = {
                'target': target_es, 'steps': steps,
                'vqm': vqm_utility, 'delay': delay, 'energy': energy  # <--- 如实返回真实的百焦耳级物理能耗
            }

        self._generate_states()
        done = self.current_step >= self.max_steps
        dones = {f"ue_{i}": done for i in range(self.num_ues)}

        return self._get_normalized_states(), rewards, dones, False, infos