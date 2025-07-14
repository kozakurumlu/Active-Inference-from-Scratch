import numpy as np
import matplotlib.pyplot as plt

from new_environment import GridWorld
from new_agent import ActiveInferenceAgent
from tqdm import tqdm

def run_single_simulation(noise_level, num_trials=500, max_steps_per_trial=25):
    env = GridWorld(grid_size=3, noise_level=noise_level)
    agent = ActiveInferenceAgent(env)

    successful_trials = []
    success_rate_history = []
    last_success_rate = 0.0

    for trial in range(num_trials):
        obs = env.reset()
        agent.reset()

        done = False
        is_success = False
        step_count = 0

        # Accuracy-based learning rate: higher when accuracy low, lower when high
        power = 3  # or 3, or higher for sharper decay
        current_lr = 0.1 * (1 - last_success_rate**power) + 0.01 * (last_success_rate**power)

        while not done and step_count < max_steps_per_trial:
            agent.infer_states(obs)
            agent.learn_from_experience(agent.q_s, obs, learning_rate=current_lr)
            action = agent.select_action()
            obs, reward, done = env.step(action)

            if reward > 0:
                is_success = True

            step_count += 1

        successful_trials.append(1 if is_success else 0)
        window_size = min(len(successful_trials), 50)
        current_success_rate = np.mean(successful_trials[-window_size:])
        success_rate_history.append(current_success_rate)

        last_success_rate = current_success_rate

    final_success_rate = np.mean(successful_trials)
    return final_success_rate


def run_experiment(noise_levels=[0.0, 0.1, 0.2, 0.3, 0.4]):
    success_rates = []
    num_runs = 5
    num_trials = 500
    max_steps_per_trial = 25
    for noise in tqdm(noise_levels, desc="Noise Levels"):
        run_successes = []
        with tqdm(total=num_runs * num_trials, desc=f"Noise {noise}", leave=False) as pbar:
            for run in range(num_runs):
                rate = run_single_simulation(noise, num_trials=num_trials, max_steps_per_trial=max_steps_per_trial)
                run_successes.append(rate)
                pbar.update(num_trials)
        avg_rate = np.mean(run_successes)
        success_rates.append(avg_rate)
        tqdm.write(f"Noise level {noise}: Average final success rate over {num_runs} runs: {avg_rate:.2%}")

    # Bar chart
    plt.bar(noise_levels, success_rates)
    plt.xlabel('Noise Level')
    plt.ylabel('Success Rate')
    plt.title('Success Rate vs Noise Level')
    plt.ylim(0, 1)
    plt.show()

if __name__ == '__main__':
    run_experiment()