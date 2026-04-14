import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def plot_experiment_results(full_results, eval_results, save_path="figures/marl_full_evaluation.png"):
    """学术级别自动绘图：四合一连环子图 (全面对标顶会规范)"""

    # 设置 seaborn 学术高定风格
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

    # 采用 1x4 的大宽幅布局，为 LVM 的 VQM 质量指标分配独立展示空间
    fig = plt.figure(figsize=(24, 6))

    # 完美扣题的总标题 (Suptitle)
    fig.suptitle("Joint Inference and Offloading Performance for MEC-Empowered LVM Services",
                 fontsize=18, fontweight='bold', y=1.05)

    algos = ['MA-Diffusion-RL', 'MAPPO', 'Greedy', 'Random', 'LocalOnly']
    display_names = ['Diffusion-RL', 'MAPPO', 'Greedy', 'Random', 'Local']

    # 采用 seaborn 的 'deep' 高级调色盘区分不同算法
    algo_palette = sns.color_palette("deep", len(algos))
    colors = {algo: algo_palette[i] for i, algo in enumerate(algos)}

    # =========================================================
    # Subplot (a): Training Convergence Curve (重点展示 Env C)
    # =========================================================
    ax1 = fig.add_subplot(1, 4, 1)
    metrics_env_C = full_results["C"]

    # 剔除未经训练的基线算法，防止由于离群平直线压缩 Y 轴视野
    rl_algos = ['MA-Diffusion-RL', 'MAPPO']

    for algo in rl_algos:
        if algo in metrics_env_C:
            rewards = metrics_env_C[algo]['reward']
            window = max(10, len(rewards) // 20)
            smoothed = np.convolve(rewards, np.ones(window) / window, mode='valid')
            x_axis = range(len(smoothed))

            ax1.plot(x_axis, smoothed, label=algo, color=colors.get(algo, 'k'), linewidth=2.5)
            std_val = np.std(smoothed[-50:]) if len(smoothed) > 50 else np.std(smoothed)
            ax1.fill_between(x_axis, smoothed - std_val * 0.3, smoothed + std_val * 0.3,
                             alpha=0.15, color=colors.get(algo, 'k'))

    ax1.set_title("(a) Training Convergence (50 UEs)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Training Episodes", fontsize=12)
    ax1.set_ylabel("System Accumulated Reward", fontsize=12)
    ax1.legend(loc='lower right', frameon=True, fontsize=11)

    # =========================================================
    # Subplot (b): Operational Costs: Latency & Energy (双Y轴)
    # =========================================================
    ax2 = fig.add_subplot(1, 4, 2)
    ax2_twin = ax2.twinx()

    x = np.arange(len(algos))
    width = 0.35

    lat_vals = [eval_results["C"][algo]['latency'] for algo in algos]
    eng_vals = [eval_results["C"][algo]['energy'] for algo in algos]

    # 针对指标使用不同色系 (Muted palette)
    color_lat = sns.color_palette("muted")[0]  # 蓝色代表延迟
    color_eng = sns.color_palette("muted")[3]  # 红色代表能耗

    bar1 = ax2.bar(x - width / 2, lat_vals, width, label='Avg Latency (s)', color=color_lat, alpha=0.85,
                   edgecolor='black')
    bar2 = ax2_twin.bar(x + width / 2, eng_vals, width, label='System Energy (J)', color=color_eng, alpha=0.85,
                        edgecolor='black')

    ax2.set_title("(b) Operational Costs Comparison", fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(display_names, fontsize=11, rotation=20)

    ax2.set_ylabel("Average Latency (s)", fontsize=12, color=color_lat)
    ax2_twin.set_ylabel("System Total Energy (J)", fontsize=12, color=color_eng)
    ax2.tick_params(axis='y', labelcolor=color_lat)
    ax2_twin.tick_params(axis='y', labelcolor=color_eng)

    # 合并图例
    lines = [bar1, bar2]
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=True, fontsize=10)
    ax2.grid(False)  # 关闭主轴网格以防双Y轴网格线交叉混乱

    # =========================================================
    # Subplot (c): LVM Quality Assessment (独立高亮 MD-VQM)
    # =========================================================
    ax3 = fig.add_subplot(1, 4, 3)

    vqm_vals = [eval_results["C"][algo]['vqm'] for algo in algos]

    # 按照算法颜色绘制 VQM 分数柱状图
    bars = ax3.bar(x, vqm_vals, 0.6, color=[colors[a] for a in algos], alpha=0.9, edgecolor='black')

    ax3.set_title("(c) LVM Quality (Avg MD-VQM)", fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(display_names, fontsize=11, rotation=20)
    ax3.set_ylabel("Avg MD-VQM Score", fontsize=12)

    # 在柱体上方添加具体数值，凸显精度
    for bar in bars:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, yval + 0.3, f"{yval:.1f}",
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    # =========================================================
    # Subplot (d): Scalability Impact (规模扩展性折线图)
    # =========================================================
    ax4 = fig.add_subplot(1, 4, 4)
    ax4_twin = ax4.twinx()

    env_sizes = ['Small\n(10 UE)', 'Medium\n(25 UE)', 'Large\n(50 UE)']
    x_scale = np.arange(len(env_sizes))

    for algo in algos:
        rwd_vals = [eval_results[env][algo]['reward'] for env in ["A", "B", "C"]]
        lat_vals = [eval_results[env][algo]['latency'] for env in ["A", "B", "C"]]

        line1, = ax4.plot(x_scale, rwd_vals, marker='o', markersize=8, linestyle='-',
                          color=colors.get(algo), linewidth=2.5)

        line2, = ax4_twin.plot(x_scale, lat_vals, marker='x', markersize=8, linestyle='--',
                               color=colors.get(algo), linewidth=2.5, alpha=0.6)

        if algo == algos[0]:
            line1.set_label("Reward (Solid)")
            line2.set_label("Latency (Dashed)")

    ax4.set_title("(d) Scalability Impact", fontsize=14, fontweight='bold')
    ax4.set_xticks(x_scale)
    ax4.set_xticklabels(env_sizes, fontsize=12)

    ax4.set_ylabel("System Total Reward", fontsize=12)
    ax4_twin.set_ylabel("Average Latency (s)", fontsize=12)

    lines, labels = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines + lines2, labels + labels2, loc='lower left', frameon=True, fontsize=10)
    ax4.grid(False)

    # 整体排版压缩与保存
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 学术图表已成功保存至路径: {save_path}")