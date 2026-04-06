import matplotlib.pyplot as plt
import numpy as np
import os


def plot_learning_curves(results_dict, save_path="figures/training_comparison.png"):
    """绘制对比学习曲线并保存"""
    plt.figure(figsize=(10, 6))

    colors = {'Diffusion-RL': '#E64B35', 'PPO': '#4DBBD5', 'Greedy': '#00A087'}

    for agent_name, rewards in results_dict.items():
        # [优化] 动态调整滑动窗口大小，以适应 1500 回合的数据量，使图表更加平滑美观
        window = max(10, len(rewards) // 30)
        smoothed_rewards = np.convolve(rewards, np.ones(window) / window, mode='valid')
        x_axis = range(len(smoothed_rewards))

        plt.plot(x_axis, smoothed_rewards, label=agent_name, color=colors.get(agent_name, 'k'), linewidth=2)

        # [优化] 置信区间阴影的统计范围也随之扩大
        std_val = np.std(smoothed_rewards[-50:]) if len(smoothed_rewards) > 50 else np.std(smoothed_rewards)
        plt.fill_between(
            x_axis,
            smoothed_rewards - std_val * 0.3,
            smoothed_rewards + std_val * 0.3,
            alpha=0.15, color=colors.get(agent_name, 'k')
        )

    plt.title("Joint Optimization of LVM Inferencing and Task Offloading in MEC", fontsize=14, fontweight='bold')
    plt.xlabel("Training Episodes", fontsize=12)
    plt.ylabel("Accumulated Reward", fontsize=12)
    plt.legend(loc='lower right', fontsize=12, frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()