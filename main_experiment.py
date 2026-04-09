import numpy as np
import torch
import os
import copy

from envs.mec_lvm_env import MultiAgentMECLVMEnvMulti
from agents.mad2rl_agent import MADiffusionRLSystem
from agents.ppo_agent import MAPPOAgentSystem
from agents.heuristic_agent import GreedyAgentSystem
from utils.plot_results import plot_experiment_results


def train_and_evaluate(agent_name, AgentClass, env, env_name, episodes=1500):
    print(f"\n🚀 开始在 Env {env_name} (规模: {env.num_ess}ES, {env.num_ues}UE) 独立训练 {agent_name} ...")

    agent = AgentClass(env.num_ues, env.obs_dim, env.action_dim)
    metrics = {"reward": [], "latency": [], "energy": [], "vqm": []}

    best_reward = -float('inf')
    best_actor_weights = copy.deepcopy(agent.actor.state_dict()) if agent_name != "Greedy" else None

    # ----------------------------------------------------
    # 第一阶段：1500轮次的主训练与参数退火
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

        if hasattr(agent, 'update_lr'):
            agent.update_lr(lr_a, lr_c)

        obs_dict, _ = env.reset()
        ep_reward, ep_latency, ep_energy, ep_vqm, steps = 0, 0, 0, 0, 0

        if hasattr(agent, 'reset_buffer'):
            agent.reset_buffer()

        while True:
            if agent_name == "Greedy":
                action_dict = agent.select_actions(obs_dict)
            elif agent_name == "MA-Diffusion-RL":
                res = agent.select_actions(obs_dict, explore=True, noise_scale=noise_scale)
                action_dict = res
            else:  # MAPPO
                res = agent.select_actions(obs_dict, explore=True)
                action_dict, log_probs = res

            next_obs_dict, rewards_dict, dones_dict, _, infos = env.step(action_dict)

            ep_reward += sum(rewards_dict.values())
            # 【修复】延迟和质量使用取均值，但系统的系统总能耗必须用 sum() 提取全场总能耗 (J)
            ep_latency += np.mean([infos[ue]['delay'] for ue in infos])
            ep_energy += np.sum([infos[ue]['energy'] for ue in infos])
            ep_vqm += np.mean([infos[ue]['vqm'] for ue in infos])
            steps += 1

            if agent_name == "MA-Diffusion-RL":
                agent.train_step(obs_dict, action_dict, rewards_dict)
            elif agent_name == "MAPPO":
                agent.store_transition(obs_dict, action_dict, log_probs, rewards_dict)

            obs_dict = next_obs_dict
            if any(dones_dict.values()):
                break

        if agent_name == "MAPPO":
            agent.update(entropy_coef=ent_coef)

        metrics["reward"].append(ep_reward)
        metrics["latency"].append(ep_latency / steps)
        metrics["energy"].append(ep_energy / steps)  # 每时隙系统总能耗
        metrics["vqm"].append(ep_vqm / steps)

        # 动态滑动窗口保存 Best Model
        if len(metrics["reward"]) >= 50:
            current_avg_rwd = np.mean(metrics["reward"][-50:])
            if current_avg_rwd > best_reward:
                best_reward = current_avg_rwd
                if agent_name != "Greedy":
                    best_actor_weights = copy.deepcopy(agent.actor.state_dict())

        if (ep + 1) % 100 == 0:
            avg_eng_print = np.mean(metrics['energy'][-50:])
            print(
                f"  └─ Ep {ep + 1}/{episodes} | Sys Rwd: {np.mean(metrics['reward'][-50:]):.2f} | Latency: {np.mean(metrics['latency'][-50:]):.3f}s | Energy: {avg_eng_print:.1f}J")

    # ----------------------------------------------------
    # 第二阶段：最终测试 (纯 Exploitation 验证最优权重)
    # ----------------------------------------------------
    print(f"  ⭐ 训练结束，正在加载 Best Model 进行 10 轮纯 Exploitation 测试...")
    if agent_name != "Greedy" and best_actor_weights is not None:
        agent.actor.load_state_dict(best_actor_weights)

    eval_metrics = {"reward": [], "latency": [], "energy": [], "vqm": []}
    for _ in range(10):
        obs_dict, _ = env.reset()
        ep_reward, ep_latency, ep_energy, ep_vqm, steps = 0, 0, 0, 0, 0
        while True:
            if agent_name == "Greedy":
                action_dict = agent.select_actions(obs_dict)
            else:
                res = agent.select_actions(obs_dict, explore=False)
                action_dict = res[0] if isinstance(res, tuple) else res

            next_obs_dict, rewards_dict, dones_dict, _, infos = env.step(action_dict)
            ep_reward += sum(rewards_dict.values())
            ep_latency += np.mean([infos[ue]['delay'] for ue in infos])
            ep_energy += np.sum([infos[ue]['energy'] for ue in infos])  # 【修复】测试阶段也要使用 sum 获取全场总能耗
            ep_vqm += np.mean([infos[ue]['vqm'] for ue in infos])
            steps += 1

            obs_dict = next_obs_dict
            if any(dones_dict.values()):
                break

        eval_metrics["reward"].append(ep_reward)
        eval_metrics["latency"].append(ep_latency / steps)
        eval_metrics["energy"].append(ep_energy / steps)
        eval_metrics["vqm"].append(ep_vqm / steps)

    final_eval_avg = {k: np.mean(v) for k, v in eval_metrics.items()}
    print(
        f"  ✅ 最终稳定表现 -> Reward: {final_eval_avg['reward']:.2f} | Latency: {final_eval_avg['latency']:.2f}s | Energy: {final_eval_avg['energy']:.1f}J")

    return metrics, final_eval_avg


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    envs = {
        "A": MultiAgentMECLVMEnvMulti(env_type="A"),
        "B": MultiAgentMECLVMEnvMulti(env_type="B"),
        "C": MultiAgentMECLVMEnvMulti(env_type="C")
    }

    algos = [
        ("MA-Diffusion-RL", MADiffusionRLSystem),
        ("MAPPO", MAPPOAgentSystem),
        ("Greedy", GreedyAgentSystem)
    ]

    full_results = {"A": {}, "B": {}, "C": {}}
    eval_results = {"A": {}, "B": {}, "C": {}}

    for env_name, env_obj in envs.items():
        print(f"\n" + "=" * 50)
        print(f"🌐 正在进入网络规模 {env_name} 独立实验组")
        print("=" * 50)
        for algo_name, AgentClass in algos:
            metrics, eval_avg = train_and_evaluate(algo_name, AgentClass, env_obj, env_name, episodes=1500)
            full_results[env_name][algo_name] = metrics
            eval_results[env_name][algo_name] = eval_avg

    print("\n✅ 所有环境下的独立训练与测试已完成！正在绘制学术排版级双Y轴长图...")
    plot_experiment_results(full_results, eval_results)