import numpy as np
import os
import sys
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from envs.mec_lvm_env import MultiAgentMECLVMEnv
from agents.marl_maddpg import MultiAgentSystem
from utils.plot_results import plot_learning_curves


def run_marl_experiment():
    print("=" * 70)
    print("🚀 启动基于 LVM 真实物理特征的多智能体(MARL)联合调度对比实验 🚀")
    print("=" * 70)

    num_ues = 3  # 3 个移动用户争抢资源
    num_ess = 2  # 2 个边缘节点提供服务

    env = MultiAgentMECLVMEnv(num_ues=num_ues, num_ess=num_ess)

    # 初始化 MADDPG 多智能体系统
    mas_system = MultiAgentSystem(num_ues, env.obs_dim, env.action_dim)

    episodes = 500
    system_rewards_history = []

    for ep in range(episodes):
        obs_dict, _ = env.reset()
        ep_system_reward = 0

        while True:
            # 分布式动作选择
            action_dict = mas_system.select_actions(obs_dict, explore=True)

            # 环境状态推进，触发资源竞争物理规律
            next_obs_dict, rewards_dict, dones_dict, _, infos = env.step(action_dict)

            # 在集中式训练中，汇总所有奖励
            step_total_reward = sum(rewards_dict.values())
            ep_system_reward += step_total_reward

            # 此处可以调用 mas_system.train_step(...)

            obs_dict = next_obs_dict

            # 只要有一个 agent done，则重置 (简化逻辑)
            if any(dones_dict.values()):
                break

        system_rewards_history.append(ep_system_reward)

        if (ep + 1) % 50 == 0:
            avg_rew = np.mean(system_rewards_history[-20:])
            print(f"Episode {ep + 1}/{episodes} | System Total QoE-Cost: {avg_rew:.2f}")
            # 打印其中一个回合的竞争细节
            print(
                f"  └─ 调度详情示例: UE_0 卸载至 ES_{infos['ue_0']['target']} (去噪步数: {infos['ue_0']['steps']}, MD-VQM: {infos['ue_0']['vqm']:.1f})")

    print("\n✅ 复杂网络拓扑与资源竞争架构已成功运行！")


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    run_marl_experiment()