import numpy as np
import os
import sys
import torch

# === 新增路径配置代码 ===
# 动态获取当前文件(main_experiment.py)所在的绝对路径，并加入系统模块搜索路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
# =========================

# 导入所有自定义的模块
from envs.mec_lvm_env import MECLVMEnv
from agents.mad2rl_agent import DiffusionRLAgent
from agents.ppo_agent import PPOAgent
from agents.heuristic_agent import GreedyAgent
from utils.plot_results import plot_learning_curves


def run_experiment():
    print("=" * 65)
    print("🚀 开始面向 MEC 场景的视觉大模型联合优化对比实验 🚀")
    print("=" * 65)

    device = "CUDA" if torch.cuda.is_available() else "CPU"
    print(f"当前使用的计算加速设备: {device}\n")

    env = MECLVMEnv()
    state_dim = env.state_dim
    action_dim = env.action_dim

    agents = {
        "Diffusion-RL": DiffusionRLAgent(state_dim, action_dim),
        "PPO": PPOAgent(state_dim, action_dim),
        "Greedy": GreedyAgent()
    }

    # [优化] 增加训练回合数至 1500，以展现中后期的长期收敛趋势
    episodes = 1500
    results = {name: [] for name in agents.keys()}

    for agent_name, agent in agents.items():
        print(f"--- 正在运行算法: [{agent_name}] ---")
        for ep in range(episodes):
            state = env.reset()
            ep_reward = 0
            while True:
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                agent.train(state, action, reward, next_state, done)

                state = next_state
                ep_reward += reward
                if done: break

            results[agent_name].append(ep_reward)

            # [优化] 将打印频率修改为每 100 回合打印一次，避免输出过多刷屏
            if (ep + 1) % 100 == 0:
                avg_reward = np.mean(results[agent_name][-20:])
                print(f"  Episode {ep + 1}/{episodes} | 最近20回合平均Reward: {avg_reward:.2f}")

        print("\n")

    print("正在处理实验数据并绘制算法收敛性能对比图表...")
    plot_learning_curves(results, save_path="figures/training_comparison.png")
    print("✅ 所有实验已圆满完成！结果图已保存至 figures/training_comparison.png。")


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    run_experiment()