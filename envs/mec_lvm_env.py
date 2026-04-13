import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MultiAgentMECLVMEnvMulti(gym.Env):
    """
    环境: 面向视觉大模型(LVM)的多智能体MEC卸载与推理步数联合优化环境。
    (INFOCOM 2025 顶会标准真实物理参数版 + 极端异构化 + 踩踏拥塞惩罚机制)
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

        # 【防拥塞重构 2】新增排队拥塞感知 queue_norm (M维)，总维度变为 3 + 4*M + 1
        self.obs_dim = 3 + 4 * self.num_ess + 1
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
        norm_states = {}
        f_norm = np.array(self.F_ess) / 80000.0
        l_norm = np.array(self.es_loads) / float(self.num_ues)

        total_used_load = sum(self.es_loads)
        bw_remain = max(0.0, 1.0 - total_used_load / (self.num_ess * self.num_ues))
        bw_norm_arr = np.array([bw_remain], dtype=np.float32)

        snr_norm_list = []
        queue_norm_list = []
        for m in range(self.num_ess):
            expected_bw = self.B_es / max(1.0, self.es_loads[m] + 1)
            snr_m = (self.P_tx * 2e-6) / (expected_bw * self.N_0)
            snr_norm_list.append(np.clip(snr_m / 1e7, 0.0, 1.0))

            # 【防拥塞重构 2】计算各服务器的预计排队拥塞程度
            queue_delay = (self.es_loads[m] * 20.0) / max(1.0, self.F_ess[m])
            queue_norm_list.append(np.clip(queue_delay * 100.0, 0.0, 1.0))

        snr_norm = np.array(snr_norm_list, dtype=np.float32)
        queue_norm = np.array(queue_norm_list, dtype=np.float32)

        for k, v in self.states.items():
            norm_v = np.copy(v)
            norm_v[0] /= 20.0
            norm_v[1] /= 100.0
            norm_v[2] /= 8e-6
            norm_states[k] = np.concatenate([norm_v, f_norm, l_norm, snr_norm, queue_norm, bw_norm_arr]).astype(
                np.float32)
        return norm_states

    def _calculate_md_vqm(self, steps):
        q_max = 20.0
        theta = 0.05
        d_min = 5
        if steps <= d_min:
            return 0.0
        return q_max * (1 - np.exp(-theta * (steps - d_min)))

    def step(self, action_dict):
        self.current_step += 1

        for agent_id, act in action_dict.items():
            if np.isnan(act).any():
                action_dict[agent_id] = np.zeros_like(act)

        offload_decisions = {}
        inference_steps = {}
        es_load_count = {k: 0 for k in range(1, self.num_ess + 1)}

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

                # 【防拥塞重构 1】废除硬截断，引入二次方级踩踏惩罚 (Stampede Penalty)
            clip_energy = min(energy, 500.0)
            T_th = 15.0

            if delay <= T_th:
                delay_penalty = 5.0 * (delay / T_th)
                stampede_penalty = 0.0
            else:
                delay_penalty = 5.0
                # 延迟每超出阈值 1 秒，将遭受剧烈的二次方爆炸惩罚，把超级节点踩踏行为彻底打痛
                stampede_penalty = 2.0 * ((delay - T_th) ** 2)

            cost = delay_penalty + 4.0 * (clip_energy / 500.0)
            reward = vqm_utility - cost - stampede_penalty

            rewards[agent_id] = reward
            infos[agent_id] = {
                'target': target_es, 'steps': steps,
                'vqm': vqm_utility, 'delay': delay, 'energy': energy
            }

        self._generate_states()
        done = self.current_step >= self.max_steps
        dones = {f"ue_{i}": done for i in range(self.num_ues)}

        return self._get_normalized_states(), rewards, dones, False, infos