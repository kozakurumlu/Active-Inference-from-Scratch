import numpy as np

class QLearningAgent:
    def __init__(self, environment):
        self.model = environment
        self.num_states = self.model.num_obs
        self.num_actions = self.model.num_actions
        self.Q = np.zeros((self.num_states, self.num_actions))
        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon = 0.1

    def reset(self):
        pass  # Q-table persists across trials for learning

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state, done):
        q_next = 0 if done else np.max(self.Q[next_state])
        self.Q[state, action] += self.alpha * (reward + self.gamma * q_next - self.Q[state, action])