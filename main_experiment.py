import numpy as np
import torch
import os
import copy
import random
import csv
import logging
import datetime

from envs.mec_lvm_env_multi import MultiAgentMECLVMEnvMulti
from agents.mad2rl_agent import MADiffusionRLSystem
from agents.ppo_agent import MAPPOAgentSystem
from agents.heuristic_agent import GreedyAgentSystem
from agents.random_agent import RandomAgentSystem
from agents.local_agent import LocalOnlyAgentSystem
from utils.plot_results import plot_experiment_results


def setup_logger():
    """配置专业级双写日志系统：输出到控制台的同时追加至日志文件"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"experiment_run_{timestamp}.log")

    logger = logging.getLogger("MEC_LVM_Experiment")
    logger.setLevel(logging.INFO)

    # 定义带时间戳的美观日志格式
    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # 配置文件 Handler
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # 配置控制台 Handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_and_evaluate(agent_name, AgentClass, env, env_name, logger, episodes=1500):
    logger.info(f"\n🚀 开始在 Env {env_name} (规模: {env.num_ess}ES, {env.num_ues}UE) 独立训练/测试 {agent_name} ...")

    agent = AgentClass(env.num_ues, env.obs_dim, env.action_dim)
    # 初始化专家导师，用于专家引导探索
    expert_agent = GreedyAgentSystem(env.num_ues, env.obs_dim, env.action_dim)

    metrics = {"reward": [], "latency": [], "energy": [], "vqm": []}

    best_reward = -float('inf')
    best_actor_weights = None
    if agent_name in ["MA-Diffusion-RL", "MAPPO"]:
        best_actor_weights = copy.deepcopy(agent.actor.state_dict())

    # ----------------------------------------------------
    # 第一阶段：1500轮次的主训练与基线采样
    # ----------------------------------------------------
    for ep in range(episodes):
        if ep < 800:
            noise_scale = 0.1
            ent_coef = 0.05
            lr_a, lr_c = 3e-4, 1e-3
        else:
            decay_ratio = (ep - 800) / 700.0
            noise_scale = 0.1 - decay_ratio * 0.09
            ent_coef = 0.05 - decay_ratio * 0.04
            lr_a = 3e-4 - decay_ratio * 2.9e-4
            lr_c = 1e-3 - decay_ratio * 9e-4

            # 专家引导概率衰减逻辑
        if ep < 500:
            expert_prob = 0.5 - (0.5 - 0.05) * (ep / 500.0)
        else:
            expert_prob = 0.05

        if hasattr(agent, 'update_lr') and agent_name in ["MA-Diffusion-RL", "MAPPO"]:
            agent.update_lr(lr_a, lr_c)

        obs_dict, _ = env.reset()
        ep_reward, ep_latency, ep_energy, ep_vqm, steps = 0, 0, 0, 0, 0

        if hasattr(agent, 'reset_buffer'):
            agent.reset_buffer()

        while True:
            # 引入 Action Mask 传入
            action_mask_dict = None
            if hasattr(env, 'get_action_mask'):
                action_mask_dict = {aid: env.get_action_mask(aid) for aid in obs_dict.keys()}

            if agent_name in ["Greedy", "Random", "LocalOnly"]:
                action_dict = agent.select_actions(obs_dict)

            elif agent_name == "MA-Diffusion-RL":
                # 【专家引导混合探索】以 expert_prob 的概率使用 Greedy，否则使用当前策略网络
                if random.random() < expert_prob:
                    action_dict = expert_agent.select_actions(obs_dict)
                else:
                    action_dict = agent.select_actions(obs_dict, explore=True, noise_scale=noise_scale,
                                                       action_mask_dict=action_mask_dict)

            else:  # MAPPO
                res = agent.select_actions(obs_dict, explore=True)
                action_dict, log_probs = res

            next_obs_dict, rewards_dict, dones_dict, _, infos = env.step(action_dict)

            ep_reward += sum(rewards_dict.values())
            ep_latency += np.mean([infos[ue]['delay'] for ue in infos])
            ep_energy += np.sum([infos[ue]['energy'] for ue in infos])
            ep_vqm += np.mean([infos[ue]['md_vqm'] for ue in infos])
            steps += 1

            if agent_name == "MA-Diffusion-RL":
                # 【离线 Buffer 收集】包含混探经验
                agent.store_transition(obs_dict, action_dict, rewards_dict, next_obs_dict)
                agent.train_step()  # 随机抽取批次学习
            elif agent_name == "MAPPO":
                agent.store_transition(obs_dict, action_dict, log_probs, rewards_dict)

            obs_dict = next_obs_dict
            if any(dones_dict.values()):
                break

        if agent_name == "MAPPO":
            agent.update(entropy_coef=ent_coef)

        metrics["reward"].append(ep_reward)
        metrics["latency"].append(ep_latency / steps)
        metrics["energy"].append(ep_energy / steps)
        metrics["vqm"].append(ep_vqm / steps)

        if len(metrics["reward"]) >= 50:
            current_avg_rwd = np.mean(metrics["reward"][-50:])
            if current_avg_rwd > best_reward:
                best_reward = current_avg_rwd
                if agent_name in ["MA-Diffusion-RL", "MAPPO"]:
                    best_actor_weights = copy.deepcopy(agent.actor.state_dict())

        if (ep + 1) % 100 == 0:
            avg_eng_print = np.mean(metrics['energy'][-50:])
            avg_vqm_print = np.mean(metrics['vqm'][-50:])
            # 增加 Avg MD-VQM 的重点打印记录
            logger.info(
                f"  └─ Ep {ep + 1}/{episodes} | Sys Rwd: {np.mean(metrics['reward'][-50:]):.2f} | Latency: {np.mean(metrics['latency'][-50:]):.3f}s | Energy: {avg_eng_print:.1f}J | Avg MD-VQM: {avg_vqm_print:.2f}")

    # ----------------------------------------------------
    # 第二阶段：最终测试 (10 轮纯评估)
    # ----------------------------------------------------
    logger.info(f"  ⭐ 训练完成，正在进行最终的无噪声验证测试...")
    if agent_name in ["MA-Diffusion-RL", "MAPPO"] and best_actor_weights is not None:
        agent.actor.load_state_dict(best_actor_weights)

    eval_metrics = {"reward": [], "latency": [], "energy": [], "vqm": []}
    for _ in range(10):
        obs_dict, _ = env.reset()
        ep_reward, ep_latency, ep_energy, ep_vqm, steps = 0, 0, 0, 0, 0
        while True:
            if hasattr(env, 'get_action_mask'):
                action_mask_dict = {aid: env.get_action_mask(aid) for aid in obs_dict.keys()}
            else:
                action_mask_dict = None

            if agent_name in ["Greedy", "Random", "LocalOnly"]:
                action_dict = agent.select_actions(obs_dict)
            elif agent_name == "MA-Diffusion-RL":
                res = agent.select_actions(obs_dict, explore=False, action_mask_dict=action_mask_dict)
                action_dict = res
            else:  # MAPPO
                res = agent.select_actions(obs_dict, explore=False)
                action_dict = res[0] if isinstance(res, tuple) else res

            next_obs_dict, rewards_dict, dones_dict, _, infos = env.step(action_dict)
            ep_reward += sum(rewards_dict.values())
            ep_latency += np.mean([infos[ue]['delay'] for ue in infos])
            ep_energy += np.sum([infos[ue]['energy'] for ue in infos])
            ep_vqm += np.mean([infos[ue]['md_vqm'] for ue in infos])
            steps += 1

            obs_dict = next_obs_dict
            if any(dones_dict.values()):
                break

        eval_metrics["reward"].append(ep_reward)
        eval_metrics["latency"].append(ep_latency / steps)
        eval_metrics["energy"].append(ep_energy / steps)
        eval_metrics["vqm"].append(ep_vqm / steps)

    final_eval_avg = {k: np.mean(v) for k, v in eval_metrics.items()}
    logger.info(
        f"  ✅ 最终稳定表现 -> Reward: {final_eval_avg['reward']:.2f} | Latency: {final_eval_avg['latency']:.2f}s | Energy: {final_eval_avg['energy']:.1f}J | Avg MD-VQM: {final_eval_avg['vqm']:.2f}")

    return metrics, final_eval_avg


if __name__ == "__main__":
    # 配置全局日志
    logger = setup_logger()
    logger.info("=" * 60)
    logger.info("🚀 毕业设计：面向视觉大模型(LVM)的推理与任务卸载联合优化")
    logger.info("=" * 60)

    set_seed(42)

    envs = {
        "A": MultiAgentMECLVMEnvMulti(env_type="A"),
        "B": MultiAgentMECLVMEnvMulti(env_type="B"),
        "C": MultiAgentMECLVMEnvMulti(env_type="C")
    }

    algos = [
        ("MA-Diffusion-RL", MADiffusionRLSystem),
        ("MAPPO", MAPPOAgentSystem),
        ("Greedy", GreedyAgentSystem),
        ("Random", RandomAgentSystem),
        ("LocalOnly", LocalOnlyAgentSystem)
    ]

    full_results = {"A": {}, "B": {}, "C": {}}
    eval_results = {"A": {}, "B": {}, "C": {}}

    for env_name, env_obj in envs.items():
        logger.info(f"\n" + "=" * 50)
        logger.info(f"🌐 正在进入网络规模 {env_name} 独立实验组")
        logger.info("=" * 50)
        for algo_name, AgentClass in algos:
            metrics, eval_avg = train_and_evaluate(algo_name, AgentClass, env_obj, env_name, logger, episodes=1500)
            full_results[env_name][algo_name] = metrics
            eval_results[env_name][algo_name] = eval_avg

    csv_file = "final_results.csv"
    with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Environment", "Algorithm", "Total Reward", "Avg Latency (s)", "System Energy (J)", "Avg MD-VQM"])

        for env_name in ["A", "B", "C"]:
            for algo_name, _ in algos:
                res = eval_results[env_name][algo_name]
                writer.writerow([
                    f"Env_{env_name}",
                    algo_name,
                    f"{res['reward']:.4f}",
                    f"{res['latency']:.4f}",
                    f"{res['energy']:.4f}",
                    f"{res['vqm']:.4f}"
                ])

    logger.info(f"\n📄 最终测试数据已成功汇总并无损导出至: {csv_file}")
    logger.info("✅ 正在绘制学术排版级双Y轴长图...")
    plot_experiment_results(full_results, eval_results)

    # 实验结束明确提示
    logger.info("\n🎉 实验圆满完成！")
    logger.info("👉 完整的运行日志和数据已保存至 logs/ 目录。")