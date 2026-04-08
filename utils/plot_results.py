import matplotlib.pyplot as plt
import numpy as np
import os


def plot_experiment_results(metrics_dict, scalability_dict, save_path="figures/marl_full_evaluation.png"):
    """学术级别自动绘图：三合一连环子图"""
    # 配置高级外观，规避找不到 serif 时报错，首选基础无衬线或者系统自带库
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

    fig = plt.figure(figsize=(18, 5))
    colors = {'MA-Diffusion-RL': '#E64B35', 'MAPPO': '#4DBBD5', 'Greedy': '#00A087'}

    # ---------------------------------------------------------
    # Subplot (a): Training Convergence Curve
    # ---------------------------------------------------------
    ax1 = fig.add_subplot(1, 3, 1)
    for algo, metrics in metrics_dict.items():
        rewards = metrics['reward']
        window = max(10, len(rewards) // 30)  # 动态平滑窗口
        smoothed = np.convolve(rewards, np.ones(window) / window, mode='valid')
        x_axis = range(len(smoothed))

        ax1.plot(x_axis, smoothed, label=algo, color=colors.get(algo, 'k'), linewidth=2.5)
        # 置信区间阴影
        std_val = np.std(smoothed[-50:]) if len(smoothed) > 50 else np.std(smoothed)
        ax1.fill_between(x_axis, smoothed - std_val * 0.3, smoothed + std_val * 0.3,
                         alpha=0.15, color=colors.get(algo, 'k'))

    ax1.set_title("(a) Training Convergence in Env B", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Training Episodes", fontsize=12)
    ax1.set_ylabel("System Accumulated Reward", fontsize=12)
    ax1.legend(loc='lower right', frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # ---------------------------------------------------------
    # Subplot (b): Performance Metrics Bar Charts
    # ---------------------------------------------------------
    ax2 = fig.add_subplot(1, 3, 2)
    labels = ['Avg Latency (s)', 'Avg Energy (J)', 'Avg MD-VQM']
    x = np.arange(len(labels))
    width = 0.25

    algos = ['MA-Diffusion-RL', 'MAPPO', 'Greedy']
    for i, algo in enumerate(algos):
        lat = np.mean(metrics_dict[algo]['latency'][-100:])
        eng = np.mean(metrics_dict[algo]['energy'][-100:])
        vqm = np.mean(metrics_dict[algo]['vqm'][-100:])
        values = [lat, eng, vqm]

        ax2.bar(x + (i - 1) * width, values, width, label=algo,
                color=colors.get(algo, 'k'), alpha=0.9, edgecolor='white')

    ax2.set_title("(b) Core Metrics Comparison", fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=12)
    ax2.legend(loc='upper right', frameon=True)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.6)

    # ---------------------------------------------------------
    # Subplot (c): Zero-shot Scalability Test
    # ---------------------------------------------------------
    ax3 = fig.add_subplot(1, 3, 3)
    env_sizes = ['2 ES\n10 UE', '5 ES\n25 UE', '10 ES\n50 UE']
    x_scale = np.arange(len(env_sizes))

    for algo in algos:
        y_vals = [scalability_dict[algo]['A'], scalability_dict[algo]['B'], scalability_dict[algo]['C']]
        ax3.plot(x_scale, y_vals, marker='o', markersize=8, label=algo,
                 color=colors.get(algo, 'k'), linewidth=2.5)

    ax3.set_title("(c) Zero-shot Scalability Test", fontsize=14, fontweight='bold')
    ax3.set_xticks(x_scale)
    ax3.set_xticklabels(env_sizes, fontsize=12)
    ax3.set_ylabel("System Total Reward", fontsize=12)
    ax3.legend(loc='upper left', frameon=True)
    ax3.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 学术图表已成功保存至路径: {save_path}")