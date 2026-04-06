import numpy as np


class GreedyAgent:
    """对比基线2: 启发式/贪心智能体"""

    def __init__(self):
        pass

    def select_action(self, state):
        bandwidth = state[3]
        if bandwidth > 50:
            return np.array([1.0, 1.0])
        else:
            return np.array([-1.0, -1.0])

    def train(self, state, action, reward, next_state, done):
        pass