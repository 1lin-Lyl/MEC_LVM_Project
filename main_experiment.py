import numpy as np
import torch
import os
import sys

from envs.mec_lvm_env import MultiAgentMECLVMEnvMulti
from agents.mad2rl_agent import MADiffusionRLSystem
from agents.ppo_agent import MAPPOAgentSystem
from agents.heuristic_agent import GreedyAgentSystem
from utils.plot_results import plot_experiment_results


def train_agent(agent_name, AgentClass, env, episodes=1500):
    print(f"\n🚀 开始在 Env B 训练 {agent_name} ...")
    agent = AgentClass(env.num_ues, env.obs_dim, env.action_dim)

    metrics = {"reward": [], "latency": [], "energy": [], "vqm": []}

    for ep in range(episodes):
        obs_dict, _ = env.reset()
        ep_reward, ep_latency, ep_energy, ep_vqm, steps = 0, 0, 0, 0, 0

        if hasattr(agent, 'reset_buffer'):
            agent.reset_buffer()

        while True:
            if agent_name == "Greedy":
                action_dict = agent.select_actions(obs_dict, env=env)
            else:
                explore = True if ep < int(episodes * 0.8) else False
                res = agent.select_actions(obs_dict, explore=explore)

                # PPO 会返回 log_probs，特殊解包处理
                if isinstance(res, tuple):
                    action_dict, log_probs = res
                else:
                    action_dict = res

            next_obs_dict, rewards_dict, dones_dict, _, infos = env.step(action_dict)

            # 数据指标跟踪
            ep_reward += sum(rewards_dict.values())
            ep_latency += np.mean([infos[ue]['delay'] for ue in infos])
            ep_energy += np.mean([infos[ue]['energy'] for ue in infos])
            ep_vqm += np.mean([infos[ue]['vqm'] for ue in infos])
            steps += 1

            # 强化学习权重更新
            if agent_name == "MA-Diffusion-RL":
                agent.train_step(obs_dict, action_dict, rewards_dict)
            elif agent_name == "MAPPO":
                agent.store_transition(obs_dict, action_dict, log_probs, rewards_dict)

            obs_dict = next_obs_dict
            if any(dones_dict.values()):
                break

        if agent_name == "MAPPO":
            agent.update()  # 仅回合并更新一次网络

        metrics["reward"].append(ep_reward)
        metrics["latency"].append(ep_latency / steps)
        metrics["energy"].append(ep_energy / steps)
        metrics["vqm"].append(ep_vqm / steps)

        if (ep + 1) % 100 == 0:
            print(
                f"  └─ Ep {ep + 1}/{episodes} | Avg Rwd: {np.mean(metrics['reward'][-50:]):.2f} | Latency: {np.mean(metrics['latency'][-50:]):.3f}s")

    return agent, metrics


def evaluate_zero_shot(agent, env, agent_name, test_ep=10):
    """Zero-shot 泛化测试，不需要 Central Critic 介入，仅用 Actor"""
    total_rewards = []
    for _ in range(test_ep):
        obs_dict, _ = env.reset()
        ep_reward = 0
        while True:
            if agent_name == "Greedy":
                action_dict = agent.select_actions(obs_dict, env=env)
            else:
                res = agent.select_actions(obs_dict, explore=False)
                action_dict = res[0] if isinstance(res, tuple) else res

            obs_dict, rewards_dict, dones_dict, _, _ = env.step(action_dict)
            ep_reward += sum(rewards_dict.values())
            if any(dones_dict.values()): break
        total_rewards.append(ep_reward)
    return np.mean(total_rewards)


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    # 1. 初始化三种网络规模
    env_A = MultiAgentMECLVMEnvMulti(env_type="A")  # 2 ES, 10 UE
    env_B = MultiAgentMECLVMEnvMulti(env_type="B")  # 5 ES, 25 UE
    env_C = MultiAgentMECLVMEnvMulti(env_type="C")  # 10 ES, 50 UE

    # 2. 依次在主环境 Env B 中进行完整训练
    algos = [
        ("MA-Diffusion-RL", MADiffusionRLSystem),
        ("MAPPO", MAPPOAgentSystem),
        ("Greedy", GreedyAgentSystem)
    ]

    full_metrics = {}
    scalability_results = {}

    for name, AgentClass in algos:
        trained_agent, metrics = train_agent(name, AgentClass, env_B, episodes=1500)
        full_metrics[name] = metrics

        print(f"\n🔄 正在对 {name} 进行 Zero-shot 规模扩展泛化测试...")
        reward_A = evaluate_zero_shot(trained_agent, env_A, name)
        reward_B = np.mean(metrics['reward'][-50:])  # 取训练后期收敛值
        reward_C = evaluate_zero_shot(trained_agent, env_C, name)

        scalability_results[name] = {"A": reward_A, "B": reward_B, "C": reward_C}
        print(f"  └─ Env A (Small): {reward_A:.2f} | Env B (Medium): {reward_B:.2f} | Env C (Large): {reward_C:.2f}")

    # 3. 数据出图
    print("\n✅ 所有实验和测试已圆满完成！正在绘制学术排版级长图...")
    plot_experiment_results(full_metrics, scalability_results)