import numpy as np


class GreedyAgentSystem:
    """ 基于启发式知识的贪心智能体 (Baseline 2) """

    def __init__(self, num_agents, obs_dim, action_dim):
        self.num_agents = num_agents
        # 反推解析用
        self.num_ess = action_dim - 2

    def select_actions(self, obs_dict, env=None, explore=False):
        actions = {}

        for agent_id, obs in obs_dict.items():
            # 严格防作弊：仅从传入的 State 解析
            data_size = obs[0] * 20.0
            local_cpu = obs[1] * 15.0
            channel_gain = obs[2] * 5e-6

            F_ess = obs[3: 3 + self.num_ess] * 250.0
            es_loads = obs[3 + self.num_ess: 3 + 2 * self.num_ess] * self.num_agents

            best_target = 0
            # 预估本地 40 步推理的延迟
            min_delay = (0.5 * data_size * 40) / (local_cpu + 1e-9)

            # 遍历尝试所有边缘节点
            for es_idx in range(self.num_ess):
                comp_load = es_loads[es_idx] + 1.0
                es_cpu = F_ess[es_idx]

                # 预估信道传输延迟
                alloc_bandwidth = 20.0 / comp_load
                snr = (0.1 * channel_gain) / (alloc_bandwidth * 1e-9)
                trans_rate = alloc_bandwidth * np.log2(1 + snr + 1e-9)
                trans_delay = data_size / trans_rate

                # 预估边缘计算延迟
                comp_delay = (0.5 * data_size * 40) / (es_cpu / comp_load)

                total_es_delay = trans_delay + comp_delay
                if total_es_delay < min_delay:
                    min_delay = total_es_delay
                    best_target = es_idx + 1  # 1-based index

            # 【核心逻辑】：构建一个能让环境正常解析出 best_target 的符合 Action Space 的向量
            act = np.full(self.num_ess + 2, -1.0, dtype=np.float32)
            act[best_target] = 1.0  # 令正确位置 Logit 值为最大，确保 np.argmax 选中它
            act[-1] = 0.5  # 映射为精确执行 40 步推理

            actions[agent_id] = act

        return actions

    def reset_buffer(self):
        pass

    def update(self):
        pass