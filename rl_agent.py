"""
Q-Learning RL agent for grid env.
- Uses o_idx as state (observable, ignores hidden context).
- Epsilon-greedy with decay, persistent Q-table.
"""

import numpy as np

class QLearningAgent:
    def __init__(self, num_obs, num_actions):
        self.num_obs = num_obs
        self.num_actions = num_actions
        self.Q = np.zeros((num_obs, num_actions))  # Q-table
        self.alpha = 0.2  # Optimized: Higher for faster learning in sparse rewards
        self.gamma = 0.99  # Discount
        self.epsilon = 1.0  # Initial explore
        self.epsilon_min = 0.05  # Optimized: Lower for better late exploitation
        self.epsilon_decay = 0.999  # Optimized: Slower for prolonged exploration in uncertainty

    def act(self, o_idx):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.num_actions)
        return np.argmax(self.Q[o_idx])

    def update(self, o_idx, action, reward, next_o_idx):
        best_next = np.max(self.Q[next_o_idx])
        self.Q[o_idx, action] += self.alpha * (reward + self.gamma * best_next - self.Q[o_idx, action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)