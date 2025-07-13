"""
Simulation for grid AcI vs RL comparison.
- Run both agents in parallel trials.
- Plot cum success rates, AcI VFE vs. RL avg Q.
"""

import numpy as np
import matplotlib.pyplot as plt
from environment import GridWorldEnv
from agent import ActiveInferenceAgent
from rl_agent import QLearningAgent

def run_simulation(num_trials=500, max_steps=500, grid_size=5):
    # Shared env params
    model = GridWorldEnv(0, grid_size)  # AcI model
    rl_agent = QLearningAgent(model.num_obs, model.num_actions)  # RL persistent

    # Metrics
    rewards_aci, vfes_aci = [], []
    rewards_rl, avg_qs_rl = [], []
    cum_rate_aci, cum_rate_rl = [], []

    for trial in range(num_trials):
        context = trial % 2
        model.true_context = context
        model.true_state = model.true_context

        # AcI run
        agent = ActiveInferenceAgent(model)
        agent.reset()
        total_reward_aci, trial_vfe, done, t = 0, 0.0, False, 0
        o_idx, _, _ = model.step(-1)
        while not done and t < max_steps:
            prev_q_s = agent.q_s.copy()
            agent.perceive(o_idx)
            trial_vfe += agent.vfe
            a = agent.act(trial)
            o_idx, reward, done = model.step(a)
            next_q_s = agent.q_s.copy()
            agent.update_parameters(prev_q_s, o_idx, a, next_q_s)
            total_reward_aci += reward
            t += 1
        rewards_aci.append(1 if total_reward_aci > 0 else 0)
        vfes_aci.append(trial_vfe / max(1, t))
        cum_rate_aci.append(np.mean(rewards_aci))

        # RL run (reset env)
        model.true_state = model.true_context
        total_reward_rl, avg_q, done, t = 0, 0.0, False, 0
        o_idx, _, _ = model.step(-1)
        while not done and t < max_steps:
            a = rl_agent.act(o_idx)
            next_o_idx, reward, done = model.step(a)
            rl_agent.update(o_idx, a, reward, next_o_idx)
            avg_q += np.mean(rl_agent.Q[o_idx])  # Track avg Q for this state
            total_reward_rl += reward
            o_idx = next_o_idx
            t += 1
        rl_agent.decay_epsilon()
        rewards_rl.append(1 if total_reward_rl > 0 else 0)
        avg_qs_rl.append(avg_q / max(1, t))
        cum_rate_rl.append(np.mean(rewards_rl))

        if (trial + 1) % 10 == 0:
            print(f"Trial {trial+1}: AcI Reward {rewards_aci[-1]}, VFE {vfes_aci[-1]:.2f} | RL Reward {rewards_rl[-1]}, Avg Q {avg_qs_rl[-1]:.2f}")

    # Plots
    fig, ax1 = plt.subplots()
    ax1.plot(cum_rate_aci, 'b-', label='AcI Cum Rate')
    ax1.plot(cum_rate_rl, 'g-', label='RL Cum Rate')
    ax1.set_ylabel('Cum Success Rate')
    ax1.set_ylim(0, 1)
    ax2 = ax1.twinx()
    ax2.plot(vfes_aci, 'r--', label='AcI VFE')
    ax2.plot(avg_qs_rl, 'm--', label='RL Avg Q')
    ax2.set_ylabel('VFE / Avg Q')
    plt.title(f'AcI vs RL in {grid_size}x{grid_size} Grid')
    plt.xlabel('Trial')
    fig.legend(loc='upper left')
    plt.show()

    print(f"AcI received reward in {100 * np.mean(rewards_aci):.1f}% of trials.")
    print(f"RL received reward in {100 * np.mean(rewards_rl):.1f}% of trials.")

if __name__ == '__main__':
    run_simulation()