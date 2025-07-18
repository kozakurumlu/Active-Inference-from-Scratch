import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from t_maze_env import TMazeEnv
from aci_agent import ActiveInferenceAgent
from rl_agent import QLearningAgent


def run_experiment(noise_levels, agent_cls, env_cls, num_trials=500):
    acc = {}
    for noise in noise_levels:
        env = env_cls(noise_level=noise)
        agent = agent_cls(env)
        successes = 0
        for _ in tqdm(range(num_trials), desc=f"{agent_cls.__name__} Noise {noise}"):
            obs = env.reset()
            agent.reset()
            done = False
            while not done:
                if isinstance(agent, QLearningAgent):
                    agent.update_context(obs)
                    state = agent.get_state()
                    action = agent.select_action(state)
                    obs, reward, done = env.step(action)
                    agent.update_context(obs)
                    next_state = agent.get_state()
                    agent.update(state, action, reward, next_state, done)
                else:
                    agent.infer_states(obs)
                    agent.learn(obs)
                    action = agent.select_action()
                    obs, reward, done = env.step(action)
            if reward > 0:
                successes += 1
        acc[noise] = successes / num_trials
    return acc


if __name__ == "__main__":
    noise_levels = [0.3, 0.4, 0.5, 0.6, 0.7]

    ai_acc = run_experiment(noise_levels, ActiveInferenceAgent, TMazeEnv)
    rl_acc = run_experiment(noise_levels, QLearningAgent, TMazeEnv)

    x = np.arange(len(noise_levels))
    width = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, [ai_acc[n] for n in noise_levels], width, label="Active Inference")
    plt.bar(x + width / 2, [rl_acc[n] for n in noise_levels], width, label="Q-Learning")
    plt.xticks(x, [str(n) for n in noise_levels])
    plt.xlabel("Noise Level")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()

    base = noise_levels[0]
    ai_drop = 100 * (ai_acc[base] - np.mean([ai_acc[n] for n in noise_levels[1:]])) / ai_acc[base]
    rl_drop = 100 * (rl_acc[base] - np.mean([rl_acc[n] for n in noise_levels[1:]])) / rl_acc[base]
    print(f"Average accuracy drop from noise {base}: AI {ai_drop:.1f}% | RL {rl_drop:.1f}%")

