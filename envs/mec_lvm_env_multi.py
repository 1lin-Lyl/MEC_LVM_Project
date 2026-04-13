import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MultiAgentMECLVMEnvMulti(gym.Env):
    """
    环境: 面向视觉大模型(LVM)的多智能体MEC卸载与推理步数联合优化环境。
    (INFOCOM 2025 顶会标准真实物理参数版 + 极端异构化 + 严格状态归一化 + 奖励截断有界化 + Action Masking)
    """

    def __init__(self, env_type="B"):
        super().__init__()

        if env_type == "A":
            self.num_ess = 2
            self.num_ues = 10
        elif env_type == "B":
            self.num_ess = 5
            self.num_ues = 25
        elif env_type == "C":
            self.num_ess = 10
            self.num_ues = 50
        else:
            raise ValueError("Unknown env_type")

        # 真实物理通信与硬件参数
        self.B_es = 20e6
        self.P_tx = 0.1
        self.N_0 = 3.98e-21
        self.eta = 20.0
        self.P_es_active = 250.0

        # 新增排队拥塞感知 queue_norm (M维)，总维度变为 3 + 4*M + 1
        self.obs_dim = 3 + 4 * self.num_ess + 1
        self.action_dim = self.num_ess + 2

        self.max_steps = 10
        self.current_step = 0
        self.F_ess = None
        self.states = {}
        self.es_loads = [0] * self.num_ess

        # ========================================================
        # 奖励归一化与有界化 (Reward Bounding) 常量设定
        # ========================================================
        self.T_th = 15.0  # 最大容忍延迟 (s)
        self.E_max = 5000.0  # 最大容忍系统能耗 (J)，规避百万焦耳的量级碾压
        self.violation_penalty = -10.0  # 固定拥塞违规惩罚 (取代无下限的二次方惩罚)

        # 奖励加权系数 (将各项指标映射至同一量级)
        self.w_vqm = 5.0  # 视频质量增益权重
        self.w_delay = 2.0  # 延迟惩罚权重
        self.w_energy = 1.0  # 能耗惩罚权重

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        self.es_loads = [0] * self.num_ess

        num_super_nodes = max(1, int(self.num_ess * 0.2))
        self.F_ess = []
        for m in range(self.num_ess):
            if m < num_super_nodes:
                self.F_ess.append(np.random.uniform(40000.0, 80000.0))
            else:
                self.F_ess.append(np.random.uniform(2000.0, 5000.0))

        np.random.shuffle(self.F_ess)

        self._generate_states()
        return self._get_normalized_states(), {}

    def _generate_states(self):
        num_weak_ues = max(1, int(self.num_ues * 0.3))
        ue_conditions = ['weak'] * num_weak_ues + ['good'] * (self.num_ues - num_weak_ues)
        np.random.shuffle(ue_conditions)

        for i in range(self.num_ues):
            if ue_conditions[i] == 'weak':
                h_n = np.random.uniform(1e-8, 5e-8)
            else:
                h_n = np.random.uniform(2e-6, 8e-6)

            self.states[f"ue_{i}"] = np.array([
                np.random.uniform(5, 20),
                np.random.uniform(50.0, 100.0),
                h_n
            ], dtype=np.float32)

    def _get_normalized_states(self):
        """
        状态空间的严格 Min-Max 归一化。
        确保所有传给 RL 网络的特征都死死锁在 [0, 1] 区间，从源头消灭梯度爆炸。
        """
        norm_states = {}

        # ES算力归一化: (F - F_min) / (F_max - F_min)
        f_norm = np.clip((np.array(self.F_ess) - 2000.0) / (80000.0 - 2000.0), 0.0, 1.0)
        # 负载归一化
        l_norm = np.clip(np.array(self.es_loads) / float(self.num_ues), 0.0, 1.0)

        total_used_load = sum(self.es_loads)
        bw_remain = np.clip(1.0 - total_used_load / (self.num_ess * self.num_ues), 0.0, 1.0)
        bw_norm_arr = np.array([bw_remain], dtype=np.float32)

        snr_norm_list = []
        queue_norm_list = []
        for m in range(self.num_ess):
            expected_bw = self.B_es / max(1.0, self.es_loads[m] + 1)
            snr_m = (self.P_tx * 2e-6) / (expected_bw * self.N_0)
            snr_norm_list.append(np.clip(snr_m / 1e7, 0.0, 1.0))

            # 服务器排队拥塞感知: 严格约束在 [0, 1] 之间
            queue_ratio = self.es_loads[m] / float(self.num_ues)
            queue_norm_list.append(np.clip(queue_ratio, 0.0, 1.0))

        snr_norm = np.array(snr_norm_list, dtype=np.float32)
        queue_norm = np.array(queue_norm_list, dtype=np.float32)

        for k, v in self.states.items():
            norm_v = np.zeros(3, dtype=np.float32)
            # UE 私有状态 Min-Max 归一化
            norm_v[0] = np.clip((v[0] - 5.0) / (20.0 - 5.0), 0.0, 1.0)  # Data size (5~20)
            norm_v[1] = np.clip((v[1] - 50.0) / (100.0 - 50.0), 0.0, 1.0)  # Local CPU (50~100)
            norm_v[2] = np.clip((v[2] - 1e-8) / (8e-6 - 1e-8), 0.0, 1.0)  # Channel gain (1e-8~8e-6)

            norm_states[k] = np.concatenate([norm_v, f_norm, l_norm, snr_norm, queue_norm, bw_norm_arr]).astype(
                np.float32)
        return norm_states

    def get_action_mask(self, ue_id):
        """
        【新增】获取指定 UE 的动作掩码 (Action Mask)。
        返回一个长度为 num_ess + 1 的一维数组，1.0 表示动作合法，0.0 表示动作非法。
        """
        mask = np.ones(self.num_ess + 1, dtype=np.float32)
        # 1. 本地执行(Local)始终合法，假设本地算力底线充足
        mask[0] = 1.0

        data_size, local_cpu, channel_gain = self.states[ue_id]

        # 预估时采用均值推理步数 30 步作为探测基准
        est_steps = 30
        req_cycles = self.eta * data_size * est_steps

        for m in range(self.num_ess):
            # 规则 A: 队列积压量达到系统总容量的 90% (严重拥塞，禁止新任务排队)
            if self.es_loads[m] >= 0.9 * self.num_ues:
                mask[m + 1] = 0.0
                continue

            # 规则 B: 预估延迟超时 (基于当前公共负载计算，若预估加入后直接超时，则禁止)
            competitors = self.es_loads[m] + 1
            alloc_bandwidth = self.B_es / competitors
            snr = (self.P_tx * channel_gain) / (alloc_bandwidth * self.N_0)
            trans_rate = (alloc_bandwidth * np.log2(1 + snr)) / 1e6  # Mbps

            trans_delay = data_size / (trans_rate + 1e-9)
            alloc_cpu = self.F_ess[m] / competitors
            comp_delay = req_cycles / alloc_cpu

            estimated_delay = trans_delay + comp_delay

            if estimated_delay > self.T_th:
                mask[m + 1] = 0.0

        return mask

    def _calculate_md_vqm(self, steps):
        q_max = 20.0
        theta = 0.05
        d_min = 5
        if steps <= d_min:
            return 0.0
        return q_max * (1 - np.exp(-theta * (steps - d_min)))

    def _calculate_reward(self, delay, energy, vqm_utility):
        """
        独立且安全的 Reward 计算模块。
        处理巨大物理差异，实施加权求和，并在最后加入坚固的硬截断。
        """
        # 1. 质量归一化 [0, 1]
        vqm_norm = vqm_utility / 20.0

        # 2. 延迟与能耗的软截断归一化 [0, 1]
        delay_norm = min(delay / self.T_th, 1.0)
        energy_norm = min(energy / self.E_max, 1.0)

        # 3. 严重违规惩罚 (如果队列溢出或极度拥塞/耗电，赋予固定负常数而非无限深渊)
        penalty = 0.0
        if delay > self.T_th or energy > self.E_max:
            penalty = self.violation_penalty

            # 4. 量纲对齐与加权求和 (保障基础 Reward 处于合理规模)
        base_reward = (self.w_vqm * vqm_norm) - (self.w_delay * delay_norm) - (self.w_energy * energy_norm)

        # 5. Reward Clipping: 最终安全垫，强制约束单步奖赏在 [-5, 5] 之间
        final_reward = np.clip(base_reward + penalty, -5.0, 5.0)
        return final_reward

    def step(self, action_dict):
        self.current_step += 1

        for agent_id, act in action_dict.items():
            if np.isnan(act).any():
                action_dict[agent_id] = np.zeros_like(act)

        offload_decisions = {}
        inference_steps = {}
        es_load_count = {k: 0 for k in range(1, self.num_ess + 1)}

        # 【新增】记录因选择被 Mask 掉的非法动作而失败的任务
        failed_tasks = set()

        for i in range(self.num_ues):
            agent_id = f"ue_{i}"
            act = action_dict[agent_id]

            target = int(np.argmax(act[:self.num_ess + 1]))

            # 【新增】动作掩码校验
            mask = self.get_action_mask(agent_id)
            if mask[target] == 0.0:
                # 动作非法，任务失败/掉线，不进入后续基站排队计数
                failed_tasks.add(agent_id)
                offload_decisions[agent_id] = target  # 依然记录原本意图，供 info 分析
            else:
                offload_decisions[agent_id] = target
                if target > 0:
                    es_load_count[target] += 1

            step_val = act[-1]
            steps = int(np.round((np.clip(step_val, -1.0, 1.0) + 1.0) / 2.0 * 40 + 10))
            inference_steps[agent_id] = np.clip(steps, 10, 50)

        self.es_loads = [es_load_count[m + 1] for m in range(self.num_ess)]

        rewards = {}
        infos = {}

        for i in range(self.num_ues):
            agent_id = f"ue_{i}"
            data_size, local_cpu, channel_gain = self.states[agent_id]

            target_es = offload_decisions[agent_id]
            steps = inference_steps[agent_id]

            # 【新增】处理非法掉线任务的极值惩罚
            if agent_id in failed_tasks:
                delay = self.T_th * 2.0  # 记录为严重超时的失败延迟
                energy = 0.0
                vqm_utility = 0.0
                reward = -5.0  # 直接赋予硬截断底线的严厉惩罚

                rewards[agent_id] = float(reward)
                infos[agent_id] = {
                    'target': target_es, 'steps': steps,
                    'vqm': vqm_utility, 'delay': delay, 'energy': energy,
                    'status': 'failed_masked'
                }
                continue  # 跳过物理计算，相当于任务已丢弃

            req_cycles = self.eta * data_size * steps
            vqm_utility = self._calculate_md_vqm(steps)

            delay = 0.0
            energy = 0.0

            if target_es == 0:
                local_cpu = max(local_cpu, 1e-9)
                delay = req_cycles / local_cpu
                p_local = 2.0 * ((local_cpu / 100.0) ** 3)
                energy = p_local * delay
            else:
                competitors = es_load_count[target_es]
                alloc_bandwidth = self.B_es / competitors
                snr = (self.P_tx * channel_gain) / (alloc_bandwidth * self.N_0)
                trans_rate = (alloc_bandwidth * np.log2(1 + snr)) / 1e6

                trans_delay = data_size / (trans_rate + 1e-9)
                trans_energy = self.P_tx * trans_delay

                alloc_cpu = self.F_ess[target_es - 1] / competitors
                comp_delay = req_cycles / alloc_cpu

                comp_energy = self.P_es_active * comp_delay

                delay = trans_delay + comp_delay
                energy = trans_energy + comp_energy

                # 调用重构后的有界化 Reward 计算模块
            reward = self._calculate_reward(delay, energy, vqm_utility)

            rewards[agent_id] = float(reward)
            infos[agent_id] = {
                'target': target_es, 'steps': steps,
                'vqm': vqm_utility, 'delay': delay, 'energy': energy,
                'status': 'success'
            }

        self._generate_states()
        done = self.current_step >= self.max_steps
        dones = {f"ue_{i}": done for i in range(self.num_ues)}

        return self._get_normalized_states(), rewards, dones, False, infos