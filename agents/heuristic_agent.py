import numpy as np


class GreedyAgentSystem:
    """ 基于启发式知识的贪心智能体 (Baseline 2) """

    def __init__(self, num_agents, obs_dim, action_dim):
        self.num_agents = num_agents

    def select_actions(self, obs_dict, env=None, explore=False):
        actions = {}

        if env is not None:
            es_loads = [0] * env.num_ess
            for i in range(len(obs_dict)):
                agent_id = f"ue_{i}"
                data_size, local_cpu, channel_gain = obs_dict[agent_id]

                best_es = 0
                best_score = local_cpu / (data_size + 1e-9)  # 本地处理基准分

                for es_idx in range(env.num_ess):
                    es_cpu = env.F_ess[es_idx]
                    comp_score = es_cpu / (es_loads[es_idx] + 1.0)
                    score = comp_score * (channel_gain * 1e5)

                    if score > best_score:
                        best_score = score
                        best_es = es_idx + 1  # 1-based index

                if best_es > 0:
                    es_loads[best_es - 1] += 1

                # 动作严格逆向映射到连续空间 [-1, 1] 保证底层 step 处理一致性
                act0 = (best_es / env.num_ess) * 2.0 - 1.0
                act1 = 0.5  # 映射到 40 步推理

                actions[agent_id] = np.array([act0, act1], dtype=np.float32)
        else:
            for aid in obs_dict.keys():
                actions[aid] = np.array([1.0, 1.0], dtype=np.float32)

        return actions

    def reset_buffer(self):
        pass

    def update(self):
        pass