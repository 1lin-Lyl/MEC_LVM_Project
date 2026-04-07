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

    num_ues = 3
    num_ess = 2
    env = MultiAgentMECLVMEnv(num_ues=num_ues, num_ess=num_ess)

    mas_system = MultiAgentSystem(num_ues, env.obs_dim, env.action_dim)
    episodes = 500

    # 记录字典，方便直接使用之前的画图工具
    results = {"MADDPG": []}

    for ep in range(episodes):
        obs_dict, _ = env.reset()
        ep_system_reward = 0

        while True:
            # 探索率衰减 (随着训练越来越确信)
            explore = True if ep < int(episodes * 0.8) else False
            action_dict = mas_system.select_actions(obs_dict, explore=explore)

            next_obs_dict, rewards_dict, dones_dict, _, infos = env.step(action_dict)

            step_total_reward = sum(rewards_dict.values())
            ep_system_reward += step_total_reward

            # 【Bug修复】调用真实的训练逻辑
            mas_system.train_step(obs_dict, action_dict, rewards_dict)

            obs_dict = next_obs_dict

            if any(dones_dict.values()):
                break

        results["MADDPG"].append(ep_system_reward)

        if (ep + 1) % 50 == 0:
            avg_rew = np.mean(results["MADDPG"][-20:])
            print(f"Episode {ep + 1}/{episodes} | System Total QoE-Cost: {avg_rew:.2f}")
            print(
                f"  └─ 调度详情示例: UE_0 卸载至 ES_{infos['ue_0']['target']} (去噪步数: {infos['ue_0']['steps']}, MD-VQM: {infos['ue_0']['vqm']:.1f})")

    print("\n✅ 训练完成！正在绘制多智能体收敛曲线...")
    plot_learning_curves(results, save_path="figures/marl_training_comparison.png")
    print("📊 图片已保存至 figures/marl_training_comparison.png")


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    run_marl_experiment()