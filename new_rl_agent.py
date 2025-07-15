import numpy as np

class QLearningAgent:
    def __init__(self, environment):
        self.model = environment
        self.num_actions = self.model.num_actions
        self.num_positions = self.model.num_positions
        self.num_cues = self.model.num_cues
        self.num_states = self.num_positions * 3  # position * (unknown=0, context0=1, context1=2)
        self.Q = np.zeros((self.num_states, self.num_actions))
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.95  # Discount factor
        self.epsilon_start = 0.5  # Starting exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Decay per trial
        self.epsilon = self.epsilon_start
        self.known_context = 0  # 0: unknown, 1: context0, 2: context1

    def reset(self):
        self.known_context = 0  # Reset to unknown at start of each trial
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_state(self):
        return self.model.agent_pos * 3 + self.known_context

    def update_context(self, obs):
        o_pos = obs // self.num_cues
        o_cue = obs % self.num_cues
        if o_pos == self.model.cue_pos and o_cue in [1, 2]:
            self.known_context = o_cue  # 1 for context0, 2 for context1

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state, done):
        q_next = 0 if done else np.max(self.Q[next_state])
        self.Q[state, action] += self.alpha * (reward + self.gamma * q_next - self.Q[state, action])