import numpy as np

class LocalOnlyAgentSystem:
    """ 对比基线4: 纯本地执行智能体 (极端保守对比基线) """
    def __init__(self, num_agents, obs_dim, action_dim):
        self.num_agents = num_agents
        self.num_ess = action_dim - 2

    def select_actions(self, obs_dict, explore=False):
        actions = {}
        for agent_id in obs_dict.keys():
            act = np.full(self.num_ess + 2, -1.0, dtype=np.float32)
            # 索引 0 置为 1.0，表示通过 Argmax 总是选中本地执行 (Local = 0)
            act[0] = 1.0
            # 最后一个维度的步数置为 -1.0，映射为 LVM 的最低步数 (10步)，节省本地算力
            act[-1] = -1.0
            actions[agent_id] = act
        return actions

    def reset_buffer(self): pass
    def update(self, *args, **kwargs): pass