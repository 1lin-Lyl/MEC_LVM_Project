import numpy as np


class GreedyAgentSystem:
    """ 基于启发式知识的贪心智能体 (Baseline 2) """

    def __init__(self, num_agents, obs_dim, action_dim):
        self.num_agents = num_agents

    def select_actions(self, obs_dict, env=None, explore=False):
        actions = {}

        # 该算法需要洞察环境的当前负载与配置
        if env is not None:
            es_loads = [0] * env.num_ess
            for i in range(len(obs_dict)):
                agent_id = f"ue_{i}"
                data_size, local_cpu, channel_gain = obs_dict[agent_id]

                best_es = 0
                best_score = local_cpu / (data_size + 1e-9)  # 本地得分

                # 遍历所有 ES 找到最高的分数组合
                for es_idx in range(env.num_ess):
                    es_cpu = env.F_ess[es_idx]
                    comp_score = es_cpu / (es_loads[es_idx] + 1.0)
                    score = comp_score * (channel_gain * 1e5)

                    if score > best_score:
                        best_score = score
                        best_es = es_idx + 1  # 1-based idx

                if best_es > 0:
                    es_loads[best_es - 1] += 1

                # 动作逆向映射到连续空间 [-1, 1]
                act0 = (best_es / env.num_ess) * 2.0 - 1.0
                # 推理步数贪心直接选择固定步长 40步，对应 act1 映射约为 0.5
                act1 = 0.5

                actions[agent_id] = np.array([act0, act1], dtype=np.float32)
        else:
            # 环境丢失时盲选，全部指向网络中枢并固定 50 步
            for aid in obs_dict.keys():
                actions[aid] = np.array([1.0, 1.0], dtype=np.float32)

        # 使得接口统一，兼容 PPO 抛出的两个返回值
        return actions

    def reset_buffer(self):
        pass

    def update(self):
        pass