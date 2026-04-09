import matplotlib.pyplot as plt
import numpy as np
import os


def plot_experiment_results(full_results, eval_results, save_path="figures/marl_full_evaluation.png"):
    """学术级别自动绘图：三合一连环子图"""
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

    fig = plt.figure(figsize=(18, 5))
    colors = {'MA-Diffusion-RL': '#E64B35', 'MAPPO': '#4DBBD5', 'Greedy': '#00A087'}
    algos = ['MA-Diffusion-RL', 'MAPPO', 'Greedy']

    # =========================================================
    # Subplot (a): Training Convergence Curve (重点展示 Env C)
    # =========================================================
    ax1 = fig.add_subplot(1, 3, 1)
    metrics_env_C = full_results["C"]

    for algo in algos:
        rewards = metrics_env_C[algo]['reward']
        window = max(10, len(rewards) // 20)
        smoothed = np.convolve(rewards, np.ones(window) / window, mode='valid')
        x_axis = range(len(smoothed))

        ax1.plot(x_axis, smoothed, label=algo, color=colors.get(algo, 'k'), linewidth=2.5)
        std_val = np.std(smoothed[-50:]) if len(smoothed) > 50 else np.std(smoothed)
        ax1.fill_between(x_axis, smoothed - std_val * 0.3, smoothed + std_val * 0.3,
                         alpha=0.15, color=colors.get(algo, 'k'))

    ax1.set_title("(a) Training Convergence in Env C (50 UEs)\n(Moving Average shown)", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Training Episodes", fontsize=12)
    ax1.set_ylabel("System Accumulated Reward", fontsize=12)
    ax1.legend(loc='lower right', frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # =========================================================
    # Subplot (b): Performance Metrics Bar Charts (基于 Checkpoint 评测数据)
    # =========================================================
    ax2 = fig.add_subplot(1, 3, 2)
    labels = ['Avg Latency (s)', 'Avg Energy (J)', 'Avg MD-VQM']
    x = np.arange(len(labels))
    width = 0.25

    for i, algo in enumerate(algos):
        lat = eval_results["C"][algo]['latency']
        eng = eval_results["C"][algo]['energy']
        vqm = eval_results["C"][algo]['vqm']
        values = [lat, eng, vqm]

        ax2.bar(x + (i - 1) * width, values, width, label=algo,
                color=colors.get(algo, 'k'), alpha=0.9, edgecolor='white')

    ax2.set_title("(b) Core Metrics Comparison (Env C)\n(Best Model Evaluated)", fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=12)
    ax2.legend(loc='upper right', frameon=True)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.6)

    # =========================================================
    # Subplot (c): Scalability Impact (基于各环境的 Checkpoint 评测数据)
    # =========================================================
    ax3 = fig.add_subplot(1, 3, 3)
    ax3_twin = ax3.twinx()

    env_sizes = ['2 ES\n10 UE', '5 ES\n25 UE', '10 ES\n50 UE']
    x_scale = np.arange(len(env_sizes))

    for algo in algos:
        # 提取各个算法在三个环境下的最终测试 Reward 与 Latency
        rwd_vals = [eval_results[env][algo]['reward'] for env in ["A", "B", "C"]]
        lat_vals = [eval_results[env][algo]['latency'] for env in ["A", "B", "C"]]

        line1, = ax3.plot(x_scale, rwd_vals, marker='o', markersize=8, linestyle='-',
                          color=colors.get(algo, 'k'), linewidth=2.5)

        line2, = ax3_twin.plot(x_scale, lat_vals, marker='x', markersize=8, linestyle='--',
                               color=colors.get(algo, 'k'), linewidth=2.5, alpha=0.7)

        if algo == algos[0]:
            line1.set_label("Reward (Solid)")
            line2.set_label("Latency (Dashed)")

    ax3.set_title("(c) Scalability Impact\n(Best Models Evaluated)", fontsize=13, fontweight='bold')
    ax3.set_xticks(x_scale)
    ax3.set_xticklabels(env_sizes, fontsize=12)

    ax3.set_ylabel("System Total Reward", fontsize=12)
    ax3_twin.set_ylabel("Average Latency (s)", fontsize=12)

    lines, labels = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines + lines2, labels + labels2, loc='upper right', frameon=True, fontsize=10)

    ax3.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 学术图表已成功保存至路径: {save_path}")