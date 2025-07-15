import numpy as np
import matplotlib.pyplot as plt

from new_environment import GridWorld
from new_agent import ActiveInferenceAgent
from new_rl_agent import QLearningAgent
from tqdm import tqdm

def run_single_simulation(noise_level, agent_class, num_trials=500, max_steps_per_trial=50):
    env = GridWorld(grid_size=3, noise_level=noise_level)
    agent = agent_class(env)

    successful_trials = []
    success_rate_history = []
    last_success_rate = 0.0

    for trial in range(num_trials):
        obs = env.reset()
        agent.reset()

        done = False
        is_success = False
        step_count = 0

        if agent_class == ActiveInferenceAgent:
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
        else:  # QLearningAgent
            state = obs
            while not done and step_count < max_steps_per_trial:
                action = agent.select_action(state)
                next_obs, reward, done = env.step(action)
                next_state = next_obs
                agent.update(state, action, reward, next_state, done)
                state = next_state

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

def run_experiment(noise_levels=[0.1, 0.2, 0.3, 0.4, 0.5]):
    success_rates_aci = []
    success_rates_rl = []
    num_runs = 5
    num_trials = 500
    max_steps_per_trial = 25
    for noise in tqdm(noise_levels, desc="Noise Levels"):
        aci_run_successes = []
        rl_run_successes = []
        with tqdm(total=num_runs * num_trials * 2, desc=f"Noise {noise}", leave=False) as pbar:
            for run in range(num_runs):
                rate_aci = run_single_simulation(noise, ActiveInferenceAgent, num_trials=num_trials, max_steps_per_trial=max_steps_per_trial)
                aci_run_successes.append(rate_aci)
                pbar.update(num_trials)
                rate_rl = run_single_simulation(noise, QLearningAgent, num_trials=num_trials, max_steps_per_trial=max_steps_per_trial)
                rl_run_successes.append(rate_rl)
                pbar.update(num_trials)
        avg_rate_aci = np.mean(aci_run_successes)
        avg_rate_rl = np.mean(rl_run_successes)
        success_rates_aci.append(avg_rate_aci)
        success_rates_rl.append(avg_rate_rl)
        tqdm.write(f"Noise level {noise}: AcI Average final success rate over {num_runs} runs: {avg_rate_aci:.2%}")
        tqdm.write(f"Noise level {noise}: RL Average final success rate over {num_runs} runs: {avg_rate_rl:.2%}")

    # Bar chart
    x = np.arange(len(noise_levels))
    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(x - width/2, success_rates_aci, width, label='AcI')
    ax.bar(x + width/2, success_rates_rl, width, label='RL')
    ax.set_xlabel('Noise Level')
    ax.set_ylabel('Success Rate')
    ax.set_title('Success Rate vs Noise Level for AcI and RL')
    ax.set_xticks(x)
    ax.set_xticklabels(noise_levels)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True)
    plt.show()

if __name__ == '__main__':
    run_experiment()