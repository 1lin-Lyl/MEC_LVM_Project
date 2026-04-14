import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MultiAgentMECLVMEnvMulti(gym.Env):
    """
    环境: 面向视觉大模型(LVM)的【推理精度】与【任务卸载】联合优化环境。
    (深度扣题版：引入 FP16/INT8/INT4 精度权衡，强化 VQM 视觉质量权重，严格状态归一化，Tanh软奖励有界化)
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
        # LVM 在最高精度(FP16)下的基准单步推理计算复杂度 (20 GFLOPs / Mbits)
        self.eta = 20.0
        self.P_es_active = 250.0

        # 状态与动作维度
        self.obs_dim = 3 + 4 * self.num_ess + 1
        # 动作空间：前 M+1 维是卸载决策(Offloading), 最后 1 维是 LVM推理精度(Inference Precision)
        self.action_dim = self.num_ess + 2

        self.max_steps = 10
        self.current_step = 0
        self.F_ess = None
        self.states = {}
        self.es_loads = [0] * self.num_ess

        # ========================================================
        # 【核心重构】奖励计算与 Tanh 软归一化防爆炸 (Soft Reward Normalization)
        # ========================================================
        self.T_th = 15.0  # 最大容忍延迟基准值 (s)，用于内部无量纲化
        self.E_max = 5000.0  # 最大容忍系统能耗基准值 (J)，用于内部无量纲化

        # 动态权重调整：在保证 VQM 为主导的同时，大幅增加对延迟和能耗的敏感度
        # 迫使智能体在画质达到满分后，依然有动力“精打细算”，杜绝“躺平”
        self.w_vqm = 10.0  # 视频质量增益权重
        self.w_delay = 8.0  # 延迟惩罚权重 (大幅提高)
        self.w_energy = 5.0  # 能耗惩罚权重 (大幅提高)

        # Tanh 软映射参数：取代 np.clip 的硬截断
        self.reward_scale = 10.0  # 将最终奖励平滑约束在 [-10, 10] 之间
        self.reward_temp = 10.0  # 温度系数，控制 Tanh 梯度的平滑宽度，防止早期梯度过早消失

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
        确保所有传给 RL 网络的特征都死死锁在 [0, 1] 区间，从源头消灭感知层的数值爆炸。
        """
        norm_states = {}

        f_norm = np.clip((np.array(self.F_ess) - 2000.0) / (80000.0 - 2000.0), 0.0, 1.0)
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

            queue_ratio = self.es_loads[m] / float(self.num_ues)
            queue_norm_list.append(np.clip(queue_ratio, 0.0, 1.0))

        snr_norm = np.array(snr_norm_list, dtype=np.float32)
        queue_norm = np.array(queue_norm_list, dtype=np.float32)

        for k, v in self.states.items():
            norm_v = np.zeros(3, dtype=np.float32)
            norm_v[0] = np.clip((v[0] - 5.0) / (20.0 - 5.0), 0.0, 1.0)
            norm_v[1] = np.clip((v[1] - 50.0) / (100.0 - 50.0), 0.0, 1.0)
            norm_v[2] = np.clip((v[2] - 1e-8) / (8e-6 - 1e-8), 0.0, 1.0)

            norm_states[k] = np.concatenate([norm_v, f_norm, l_norm, snr_norm, queue_norm, bw_norm_arr]).astype(
                np.float32)
        return norm_states

    def get_action_mask(self, ue_id):
        """
        获取指定 UE 的动作掩码 (Action Mask)。
        """
        mask = np.ones(self.num_ess + 1, dtype=np.float32)
        mask[0] = 1.0

        data_size, local_cpu, channel_gain = self.states[ue_id]

        # 预估时采用 INT8 量化精度 (Bit-width=8) 的计算量作为探测基准
        est_precision = 8
        req_cycles = self.eta * data_size * (est_precision / 16.0)

        for m in range(self.num_ess):
            if self.es_loads[m] >= 0.9 * self.num_ues:
                mask[m + 1] = 0.0
                continue

            competitors = self.es_loads[m] + 1
            alloc_bandwidth = self.B_es / competitors
            snr = (self.P_tx * channel_gain) / (alloc_bandwidth * self.N_0)
            trans_rate = (alloc_bandwidth * np.log2(1 + snr)) / 1e6

            trans_delay = data_size / (trans_rate + 1e-9)
            alloc_cpu = self.F_ess[m] / competitors
            comp_delay = req_cycles / alloc_cpu

            estimated_delay = trans_delay + comp_delay

            if estimated_delay > self.T_th:
                mask[m + 1] = 0.0

        return mask

    def _calculate_md_vqm(self, precision):
        """
        【联合优化】: 基于 LVM 量化级别的多维视频质量模型 (MD-VQM)
        """
        q_max = 20.0
        theta = 0.2
        d_min = 2.0
        if precision <= d_min:
            return 0.0
        return q_max * (1 - np.exp(-theta * (precision - d_min)))

    def _calculate_reward(self, delay, energy, vqm_utility):
        """
        【彻底修复】：使用 Tanh 软映射取代所有硬截断。
        保证梯度单调递增，让 RL 无论处在多差的境地，都有动力去优化哪怕 1 秒的延迟。
        """
        # 1. 内部无量纲化：提取物理特征的相对比例 (绝不使用 min/max 进行硬截断)
        vqm_norm = vqm_utility / 20.0
        delay_norm = delay / self.T_th
        energy_norm = energy / self.E_max

        # 2. 连续博弈公式：服务效用 - 资源成本惩罚
        raw_reward = (self.w_vqm * vqm_norm) - (self.w_delay * delay_norm) - (self.w_energy * energy_norm)

        # 3. Tanh 软归一化防爆炸
        # 通过 np.tanh 将极端恶劣或极度优秀的 Raw_Reward 平滑压扁至 [-SCALE, SCALE]
        # 由于 tanh 处处可导，Actor 能时刻感知到 "降低 delay_norm" 带来的微小收益
        final_reward = self.reward_scale * np.tanh(raw_reward / self.reward_temp)

        return final_reward

    def step(self, action_dict):
        self.current_step += 1

        for agent_id, act in action_dict.items():
            if np.isnan(act).any():
                action_dict[agent_id] = np.zeros_like(act)

        offload_decisions = {}
        inference_precisions = {}
        es_load_count = {k: 0 for k in range(1, self.num_ess + 1)}

        failed_tasks = set()

        for i in range(self.num_ues):
            agent_id = f"ue_{i}"
            act = action_dict[agent_id]

            target = int(np.argmax(act[:self.num_ess + 1]))

            mask = self.get_action_mask(agent_id)
            if mask[target] == 0.0:
                failed_tasks.add(agent_id)
                offload_decisions[agent_id] = target
            else:
                offload_decisions[agent_id] = target
                if target > 0:
                    es_load_count[target] += 1

            precision_levels = [4.0, 8.0, 16.0]
            precision_val = act[-1]
            idx = int(np.round((np.clip(precision_val, -1.0, 1.0) + 1.0) / 2.0 * 2))
            inference_precisions[agent_id] = precision_levels[idx]

        self.es_loads = [es_load_count[m + 1] for m in range(self.num_ess)]

        rewards = {}
        infos = {}

        for i in range(self.num_ues):
            agent_id = f"ue_{i}"
            data_size, local_cpu, channel_gain = self.states[agent_id]

            target_es = offload_decisions[agent_id]
            precision = inference_precisions[agent_id]

            if agent_id in failed_tasks:
                # 若选择了被 Mask 掉的非法动作，给予合理的虚构极端惩罚值
                delay = self.T_th * 2.0
                energy = self.E_max  # 取最大能耗进行惩罚
                vqm_utility = 0.0

                # 【统一接口】直接复用连续加权公式，不再生硬指定 -5.0
                reward = self._calculate_reward(delay, energy, vqm_utility)

                rewards[agent_id] = float(reward)
                infos[agent_id] = {
                    'target': target_es,
                    'inference_precision': precision,
                    'md_vqm': vqm_utility,
                    'delay': delay,
                    'energy': energy,
                    'status': 'failed_masked'
                }
                continue

            req_cycles = self.eta * data_size * (precision / 16.0)
            vqm_utility = self._calculate_md_vqm(precision)

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

                # 调用彻底重构后的 Tanh 软归一化防爆炸 Reward 函数
            reward = self._calculate_reward(delay, energy, vqm_utility)

            rewards[agent_id] = float(reward)
            infos[agent_id] = {
                'target': target_es,
                'inference_precision': precision,
                'md_vqm': vqm_utility,
                'delay': delay,
                'energy': energy,
                'status': 'success'
            }

        self._generate_states()
        done = self.current_step >= self.max_steps
        dones = {f"ue_{i}": done for i in range(self.num_ues)}

        return self._get_normalized_states(), rewards, dones, False, infos