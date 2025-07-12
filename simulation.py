"""
Simulation for grid AcI.
- Run episodes, track reward/VFE.
- Plot learning curves.
"""

import numpy as np
import matplotlib.pyplot as plt
from environment import GridWorldEnv
from agent import ActiveInferenceAgent

def run_simulation(num_trials=100, max_steps=500, grid_size=10):
    rewards, vfes = [], []
    cum_reward_rate = []
    model = GridWorldEnv(0, grid_size)  # Shared
    for trial in range(num_trials):
        context = trial % 2
        model.true_context = context
        model.true_state = model.true_context
        agent = ActiveInferenceAgent(model)
        agent.reset()
        total_reward, trial_vfe, done, t = 0, 0.0, False, 0
        o_idx, _, _ = model.step(-1)
        while not done and t < max_steps:
            prev_q_s = agent.q_s.copy()
            agent.perceive(o_idx)
            trial_vfe += agent.vfe
            a = agent.act()
            o_idx, reward, done = model.step(a)
            next_q_s = agent.q_s.copy()
            agent.update_parameters(prev_q_s, o_idx, a, next_q_s)
            total_reward += reward
            t += 1
        rewards.append(1 if total_reward > 0 else 0)
        vfes.append(trial_vfe / max(1, t))
        cum_reward_rate.append(np.mean(rewards))
        if (trial + 1) % 10 == 0:
            print(f"Trial {trial+1}: Reward {rewards[-1]}, Avg VFE {vfes[-1]:.2f}")

    fig, ax1 = plt.subplots()
    ax1.plot(cum_reward_rate, 'b-', label='Cum Reward Rate')
    ax1.set_ylabel('Reward Rate')
    ax1.set_ylim(0, 1)
    ax2 = ax1.twinx()
    ax2.plot(vfes, 'r--', label='Avg VFE')
    ax2.set_ylabel('VFE')
    plt.title(f'AcI in {grid_size}x{grid_size} Grid with Obstacles')
    plt.xlabel('Trial')
    fig.legend(loc='upper left')
    plt.show()

if __name__ == '__main__':
    run_simulation(num_trials=100, max_steps=500, grid_size=5)