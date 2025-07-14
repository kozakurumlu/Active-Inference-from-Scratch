import numpy as np
import matplotlib.pyplot as plt

from new_environment import GridWorld
from new_agent import ActiveInferenceAgent

def run_simulation(num_trials=500, max_steps_per_trial=25):
    env = GridWorld(grid_size=3)
    agent = ActiveInferenceAgent(env)

    successful_trials = []
    success_rate_history = []

    print("Starting simulation...")
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

        if (trial + 1) % 10 == 0:
            print(f"Trial {trial + 1}/{num_trials} | Recent Success Rate: {current_success_rate:.2f}")

    # --- Plotting ---
    plt.figure(figsize=(10, 5))
    plt.plot(success_rate_history, label='Success Rate (50-trial moving average)')
    plt.title('Active Inference Agent Learning Progress')
    plt.xlabel('Trial Number')
    plt.ylabel('Success Rate')
    plt.ylim(0, 1.1)
    plt.grid(True)
    plt.legend()
    plt.show()

    print(f"\nFinal overall success rate: {np.mean(successful_trials):.2%}")

if __name__ == '__main__':
    run_simulation()