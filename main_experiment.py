import numpy as np
import torch
import os

from envs.mec_lvm_env import MultiAgentMECLVMEnvMulti
from agents.mad2rl_agent import MADiffusionRLSystem
from agents.ppo_agent import MAPPOAgentSystem
from agents.heuristic_agent import GreedyAgentSystem
from utils.plot_results import plot_experiment_results


def train_agent(agent_name, AgentClass, env, env_name, episodes=1500):
    print(f"\n🚀 开始在 Env {env_name} (规模: {env.num_ess}ES, {env.num_ues}UE) 独立训练 {agent_name} ...")

    # 动态适应环境提供的观测维度和动作维度
    agent = AgentClass(env.num_ues, env.obs_dim, env.action_dim)
    metrics = {"reward": [], "latency": [], "energy": [], "vqm": []}

    for ep in range(episodes):
        obs_dict, _ = env.reset()
        ep_reward, ep_latency, ep_energy, ep_vqm, steps = 0, 0, 0, 0, 0

        if hasattr(agent, 'reset_buffer'):
            agent.reset_buffer()

        while True:
            if agent_name == "Greedy":
                action_dict = agent.select_actions(obs_dict)
            else:
                explore = True if ep < int(episodes * 0.8) else False
                res = agent.select_actions(obs_dict, explore=explore)

                if isinstance(res, tuple):
                    action_dict, log_probs = res
                else:
                    action_dict = res

            next_obs_dict, rewards_dict, dones_dict, _, infos = env.step(action_dict)

            ep_reward += sum(rewards_dict.values())
            ep_latency += np.mean([infos[ue]['delay'] for ue in infos])
            ep_energy += np.mean([infos[ue]['energy'] for ue in infos])
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
            agent.update()

        metrics["reward"].append(ep_reward)
        metrics["latency"].append(ep_latency / steps)
        metrics["energy"].append(ep_energy / steps)
        metrics["vqm"].append(ep_vqm / steps)

        if (ep + 1) % 100 == 0:
            print(
                f"  └─ Ep {ep + 1}/{episodes} | Sys Rwd: {np.mean(metrics['reward'][-50:]):.2f} | Latency: {np.mean(metrics['latency'][-50:]):.3f}s")

    return metrics


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    # 1. 实例化三种真实规模的环境
    envs = {
        "A": MultiAgentMECLVMEnvMulti(env_type="A"),  # 2 ES, 10 UE
        "B": MultiAgentMECLVMEnvMulti(env_type="B"),  # 5 ES, 25 UE
        "C": MultiAgentMECLVMEnvMulti(env_type="C")  # 10 ES, 50 UE
    }

    algos = [
        ("MA-Diffusion-RL", MADiffusionRLSystem),
        ("MAPPO", MAPPOAgentSystem),
        ("Greedy", GreedyAgentSystem)
    ]

    # 全局结果大字典
    full_results = {"A": {}, "B": {}, "C": {}}

    # 2. 在每个环境中进行独立严谨的从头训练
    for env_name, env_obj in envs.items():
        print(f"\n" + "=" * 50)
        print(f"🌐 正在进入网络规模 {env_name} 独立实验组")
        print("=" * 50)
        for algo_name, AgentClass in algos:
            metrics = train_agent(algo_name, AgentClass, env_obj, env_name, episodes=1500)
            full_results[env_name][algo_name] = metrics

    # 3. 数据出图
    print("\n✅ 所有环境下的独立训练实验已圆满完成！正在绘制学术排版级长图...")
    plot_experiment_results(full_results)