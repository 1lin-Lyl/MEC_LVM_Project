import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MultiAgentMECLVMEnv(gym.Env):
    """
    V2升级版：多智能体、多边缘节点、基于真实扩散模型(Diffusion)物理特征的MEC联合优化环境
    """

    def __init__(self, num_ues=3, num_ess=2):
        super(MultiAgentMECLVMEnv, self).__init__()

        self.num_ues = num_ues
        self.num_ess = num_ess

        # 物理常量定义
        self.B_total = 100.0  # 总基站带宽 (MHz)
        self.P_tx = 0.1  # UE发送功率 (W)
        self.N_0 = 1e-9  # 噪声功率
        self.eta = 0.5  # LVM 复杂度系数

        self.obs_dim = 3
        self.action_dim = 2

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
        self.F_ess = [np.random.uniform(150, 250) for _ in range(self.num_ess)]

        self.states = {}
        for i in range(self.num_ues):
            self.states[f"ue_{i}"] = np.array([
                np.random.uniform(5, 20),  # S_n
                np.random.uniform(5, 15),  # F_loc
                np.random.uniform(1e-6, 5e-6)  # h_n
            ], dtype=np.float32)

        return self._get_normalized_states(), {}

    def _get_normalized_states(self):
        norm_states = {}
        for k, v in self.states.items():
            norm_v = np.copy(v)
            norm_v[0] /= 20.0
            norm_v[1] /= 15.0
            norm_v[2] /= 5e-6
            norm_states[k] = norm_v
        return norm_states

    def _calculate_md_vqm(self, steps):
        q_max = 100.0
        theta = 0.05
        d_min = 5
        if steps <= d_min:
            return 0.0
        return q_max * (1 - np.exp(-theta * (steps - d_min)))

    def step(self, action_dict):
        self.current_step += 1

        offload_decisions = {}
        inference_steps = {}
        es_load_count = {k: 0 for k in range(1, self.num_ess + 1)}

        for i in range(self.num_ues):
            agent_id = f"ue_{i}"
            act = action_dict[agent_id]

            target = int((np.clip(act[0], -1.0, 1.0) + 1.0) / 2.0 * self.num_ess)
            target = min(target, self.num_ess)
            offload_decisions[agent_id] = target

            if target > 0:
                es_load_count[target] += 1

            steps = int(45 * np.clip(act[1], -1.0, 1.0) + 55)
            inference_steps[agent_id] = steps

        rewards = {}
        infos = {}

        for i in range(self.num_ues):
            agent_id = f"ue_{i}"
            state = self.states[agent_id]
            data_size, local_cpu, channel_gain = state

            target_es = offload_decisions[agent_id]
            steps = inference_steps[agent_id]

            req_cycles = self.eta * data_size * steps
            vqm_utility = self._calculate_md_vqm(steps)

            delay = 0.0
            energy = 0.0

            if target_es == 0:
                delay = req_cycles / local_cpu
                energy = 0.5 * data_size * (local_cpu ** 2) * (steps / 100.0)
            else:
                competitors = es_load_count[target_es]
                alloc_bandwidth = self.B_total / competitors
                snr = (self.P_tx * channel_gain) / (alloc_bandwidth * self.N_0)
                trans_rate = alloc_bandwidth * np.log2(1 + snr)

                trans_delay = data_size / trans_rate
                trans_energy = self.P_tx * trans_delay

                alloc_cpu = self.F_ess[target_es - 1] / competitors
                comp_delay = req_cycles / alloc_cpu

                delay = trans_delay + comp_delay
                energy = trans_energy

            cost = 1.2 * delay + 0.5 * energy
            reward = vqm_utility - cost
            rewards[agent_id] = reward
            infos[agent_id] = {'target': target_es, 'steps': steps, 'vqm': vqm_utility, 'delay': delay}

        # 【Bug修复区】不要调用 self.reset()！仅动态更新状态属性。
        for i in range(self.num_ues):
            self.states[f"ue_{i}"] = np.array([
                np.random.uniform(5, 20),
                np.random.uniform(5, 15),
                np.random.uniform(1e-6, 5e-6)  # 每步独立同分布更新信道
            ], dtype=np.float32)

        done = self.current_step >= self.max_steps
        dones = {f"ue_{i}": done for i in range(self.num_ues)}

        return self._get_normalized_states(), rewards, dones, False, infos