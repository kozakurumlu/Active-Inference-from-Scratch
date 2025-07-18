import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from t_maze_env import TMazeEnv
from aci_agent import ActiveInferenceAgent
from rl_agent import QLearningAgent


def run_experiment(noise_levels, agent_cls, num_trials=1000):
    histories = {}
    for noise in noise_levels:
        env = TMazeEnv(noise_level=noise)
        agent = agent_cls(env)
        successes = []
        history = []
        for _ in tqdm(range(num_trials), desc=f"Noise {noise}"):
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
            successes.append(1 if reward > 0 else 0)
            history.append(np.mean(successes))
        histories[noise] = history
    return histories


if __name__ == "__main__":
    noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    # Active Inference agent
    ai_hist = run_experiment(noise_levels, ActiveInferenceAgent)
    x = np.arange(len(next(iter(ai_hist.values()))))
    plt.figure(figsize=(8, 5))
    for noise, hist in ai_hist.items():
        plt.plot(x, hist, label=f"AI Noise {noise}")
    plt.xlabel("Trial")
    plt.ylabel("Cumulative Success Rate")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Q-learning agent
    rl_hist = run_experiment(noise_levels, QLearningAgent)
    x = np.arange(len(next(iter(rl_hist.values()))))
    plt.figure(figsize=(8, 5))
    for noise, hist in rl_hist.items():
        plt.plot(x, hist, label=f"QL Noise {noise}")
    plt.xlabel("Trial")
    plt.ylabel("Cumulative Success Rate")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()