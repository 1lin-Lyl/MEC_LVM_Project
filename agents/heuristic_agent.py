import numpy as np


class GreedyAgentSystem:
    """ 对比基线2: 基于启发式知识的贪心智能体 (彻底剥夺上帝视角版) """

    def __init__(self, num_agents, obs_dim, action_dim):
        self.num_agents = num_agents
        # 动态解析 ES 数量 (obs_dim = 3 + 4*M + 1 -> 4M = obs_dim - 4)
        self.num_ess = (obs_dim - 4) // 4

    def select_actions(self, obs_dict, explore=False):
        actions = {}

        for agent_id, obs in obs_dict.items():
            data_size = obs[0] * 20.0
            local_cpu = obs[1] * 100.0
            channel_gain = obs[2] * 8e-6

            F_ess = obs[3: 3 + self.num_ess] * 80000.0

            # 【防拥塞重构 3】严格只读取进入 Step 前的共享静态负载状态，绝不动态更新
            # 如果超级节点账面算力过大，即使当前有负载，所有独立决策的 UE 也可能认为它最好，从而导致集中踩踏
            es_loads = obs[3 + self.num_ess: 3 + 2 * self.num_ess] * self.num_agents

            best_target = 0
            min_delay = (0.5 * data_size * 40) / (local_cpu + 1e-9)

            for es_idx in range(self.num_ess):
                # 假设自己加入后的盲目负载预估
                comp_load = es_loads[es_idx] + 1.0
                es_cpu = F_ess[es_idx]

                alloc_bandwidth = 20e6 / comp_load
                snr = (0.1 * channel_gain) / (alloc_bandwidth * 3.98e-21)
                trans_rate = (alloc_bandwidth * np.log2(1 + snr)) / 1e6
                trans_delay = data_size / (trans_rate + 1e-9)

                comp_delay = (20.0 * data_size * 40) / (es_cpu / comp_load)

                total_es_delay = trans_delay + comp_delay
                if total_es_delay < min_delay:
                    min_delay = total_es_delay
                    best_target = es_idx + 1  # 1-based index

            act = np.full(self.num_ess + 2, -1.0, dtype=np.float32)
            act[best_target] = 1.0
            act[-1] = 0.5  # 固定约 40 步推理

            actions[agent_id] = act

        return actions

    def reset_buffer(self):
        pass

    def update(self, *args, **kwargs):
        pass