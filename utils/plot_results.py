import matplotlib.pyplot as plt
import numpy as np
import os


def plot_experiment_results(full_results, save_path="figures/marl_full_evaluation.png"):
    """学术级别自动绘图：三合一连环子图"""
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

    fig = plt.figure(figsize=(18, 5))
    colors = {'MA-Diffusion-RL': '#E64B35', 'MAPPO': '#4DBBD5', 'Greedy': '#00A087'}
    algos = ['MA-Diffusion-RL', 'MAPPO', 'Greedy']

    # =========================================================
    # Subplot (a): Training Convergence Curve (Only for Env B)
    # =========================================================
    ax1 = fig.add_subplot(1, 3, 1)
    metrics_env_B = full_results["B"]

    for algo in algos:
        rewards = metrics_env_B[algo]['reward']
        window = max(10, len(rewards) // 30)
        smoothed = np.convolve(rewards, np.ones(window) / window, mode='valid')
        x_axis = range(len(smoothed))

        ax1.plot(x_axis, smoothed, label=algo, color=colors.get(algo, 'k'), linewidth=2.5)
        std_val = np.std(smoothed[-50:]) if len(smoothed) > 50 else np.std(smoothed)
        ax1.fill_between(x_axis, smoothed - std_val * 0.3, smoothed + std_val * 0.3,
                         alpha=0.15, color=colors.get(algo, 'k'))

    ax1.set_title("(a) Training Convergence in Env B", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Training Episodes", fontsize=12)
    ax1.set_ylabel("System Accumulated Reward", fontsize=12)
    ax1.legend(loc='lower right', frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # =========================================================
    # Subplot (b): Performance Metrics Bar Charts (Env B 收敛后)
    # =========================================================
    ax2 = fig.add_subplot(1, 3, 2)
    labels = ['Avg Latency (s)', 'Avg Energy (J)', 'Avg MD-VQM']
    x = np.arange(len(labels))
    width = 0.25

    for i, algo in enumerate(algos):
        lat = np.mean(metrics_env_B[algo]['latency'][-100:])
        eng = np.mean(metrics_env_B[algo]['energy'][-100:])
        vqm = np.mean(metrics_env_B[algo]['vqm'][-100:])
        values = [lat, eng, vqm]

        ax2.bar(x + (i - 1) * width, values, width, label=algo,
                color=colors.get(algo, 'k'), alpha=0.9, edgecolor='white')

    ax2.set_title("(b) Core Metrics Comparison (Env B)", fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=12)
    ax2.legend(loc='upper left', frameon=True)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.6)

    # =========================================================
    # Subplot (c): Scalability Impact (独立训练收敛结果)
    # =========================================================
    ax3 = fig.add_subplot(1, 3, 3)
    ax3_twin = ax3.twinx()  # 启用双Y轴

    env_sizes = ['2 ES\n10 UE', '5 ES\n25 UE', '10 ES\n50 UE']
    x_scale = np.arange(len(env_sizes))

    for algo in algos:
        # 提取每个环境最后 100 轮收敛后的均值
        rwd_vals = [np.mean(full_results[env][algo]['reward'][-100:]) for env in ["A", "B", "C"]]
        lat_vals = [np.mean(full_results[env][algo]['latency'][-100:]) for env in ["A", "B", "C"]]

        # 绘制 Reward 曲线 (左Y轴, 实线)
        line1, = ax3.plot(x_scale, rwd_vals, marker='o', markersize=8, linestyle='-',
                          color=colors.get(algo, 'k'), linewidth=2.5)

        # 绘制 Latency 曲线 (右Y轴, 虚线加叉)
        line2, = ax3_twin.plot(x_scale, lat_vals, marker='x', markersize=8, linestyle='--',
                               color=colors.get(algo, 'k'), linewidth=2.5, alpha=0.7)

        # 只在第一次循环时添加图例的代理Artist
        if algo == algos[0]:
            line1.set_label("Reward (Solid)")
            line2.set_label("Latency (Dashed)")

    ax3.set_title("(c) Scalability Impact (Independent Training)", fontsize=14, fontweight='bold')
    ax3.set_xticks(x_scale)
    ax3.set_xticklabels(env_sizes, fontsize=12)

    ax3.set_ylabel("System Total Reward", fontsize=12)
    ax3_twin.set_ylabel("Average Latency (s)", fontsize=12)

    # 合并两个轴的图例
    lines, labels = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines + lines2, labels + labels2, loc='upper right', frameon=True, fontsize=10)

    ax3.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 学术图表已成功保存至路径: {save_path}")