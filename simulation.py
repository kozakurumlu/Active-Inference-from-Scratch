import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from t_maze_env import TMazeEnv
from agent_models import ActiveInferenceAgent


def run_experiment(noise_levels, num_trials=500):
    histories = {}
    for noise in noise_levels:
        env = TMazeEnv(noise_level=noise)
        agent = ActiveInferenceAgent(env)
        successes = []
        history = []
        for _ in tqdm(range(num_trials), desc=f"Noise {noise}"):
            obs = env.reset()
            agent.reset()
            done = False
            while not done:
                agent.infer_states(obs)
                agent.learn(obs)
                action = agent.select_action()
                obs, reward, done = env.step(action)
            successes.append(1 if reward > 0 else 0)
            history.append(np.mean(successes))
        histories[noise] = history
    return histories


if __name__ == "__main__":
    noise_levels = [0.1, 0.2, 0.3,0.4, 0.5]
    histories = run_experiment(noise_levels)
    x = np.arange(len(next(iter(histories.values()))))
    plt.figure(figsize=(8, 5))
    for noise, hist in histories.items():
        plt.plot(x, hist, label=f"Noise {noise}")
    plt.xlabel("Trial")
    plt.ylabel("Cumulative Success Rate")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

