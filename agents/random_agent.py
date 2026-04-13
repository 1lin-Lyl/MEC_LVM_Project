import numpy as np

class RandomAgentSystem:
    """ 对比基线3: 纯随机动作智能体 (系统性能下界指示器) """
    def __init__(self, num_agents, obs_dim, action_dim):
        self.num_agents = num_agents
        self.num_ess = action_dim - 2

    def select_actions(self, obs_dict, explore=False):
        actions = {}
        for agent_id in obs_dict.keys():
            # 生成随机连续动作 [-1, 1]，环境会在 step 时自行映射解析
            act = np.random.uniform(-1.0, 1.0, size=(self.num_ess + 2,))
            actions[agent_id] = np.array(act, dtype=np.float32)
        return actions

    def reset_buffer(self): pass
    def update(self, *args, **kwargs): pass