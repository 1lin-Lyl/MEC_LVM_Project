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
    # Subplot (b): Performance Metrics Bar Charts (双 Y 轴重构)
    # =========================================================
    ax2 = fig.add_subplot(1, 3, 2)
    ax2_twin = ax2.twinx()  # 分配独立 Y 轴给庞大的焦耳级别能耗数据

    x = np.arange(len(algos))
    width = 0.25

    lat_vals = [eval_results["C"][algo]['latency'] for algo in algos]
    eng_vals = [eval_results["C"][algo]['energy'] for algo in algos]
    vqm_vals = [eval_results["C"][algo]['vqm'] for algo in algos]

    # 将 Latency 和 VQM 画在左轴，System Energy 画在右轴
    ax2.bar(x - width, lat_vals, width, label='Avg Latency (s)', color='#4DBBD5', alpha=0.9, edgecolor='white')
    ax2_twin.bar(x, eng_vals, width, label='System Energy (J)', color='#E64B35', alpha=0.9, edgecolor='white')
    ax2.bar(x + width, vqm_vals, width, label='Avg MD-VQM Score', color='#00A087', alpha=0.9, edgecolor='white')

    ax2.set_title("(b) Core Metrics Comparison (Env C)\n(Best Model Evaluated)", fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(algos, fontsize=12)
    ax2.set_ylabel("Latency (s) / VQM Score", fontsize=12)
    ax2_twin.set_ylabel("System Total Energy (J)", fontsize=12)

    # 合并图例，悬浮放在不遮挡的上方位置
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=True,
               fontsize=10)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.6)

    # =========================================================
    # Subplot (c): Scalability Impact (基于各环境的 Checkpoint 评测数据)
    # =========================================================
    ax3 = fig.add_subplot(1, 3, 3)
    ax3_twin = ax3.twinx()

    env_sizes = ['2 ES\n10 UE', '5 ES\n25 UE', '10 ES\n50 UE']
    x_scale = np.arange(len(env_sizes))

    for algo in algos:
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
    ax3.legend(lines + lines2, labels + labels2, loc='upper left', frameon=True, fontsize=10)

    ax3.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 学术图表已成功保存至路径: {save_path}")