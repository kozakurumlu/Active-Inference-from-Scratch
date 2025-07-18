import numpy as np

class QLearningAgent:
    """Simple Q-learning agent for the T-Maze environment."""

    def __init__(self, environment, alpha=0.1, gamma=0.95):
        self.env = environment
        self.num_actions = self.env.num_actions
        self.num_positions = self.env.num_positions
        self.num_cues = self.env.num_cues

        # state = (position, context_estimate) with context_estimate \in {0,1,2}
        self.num_states = self.num_positions * 3
        self.Q = np.zeros((self.num_states, self.num_actions))

        self.alpha = alpha
        self.gamma = gamma

        self.epsilon_start = 0.5
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.epsilon = self.epsilon_start

        self.context_est = 0  # 0: unknown, 1: left, 2: right

    def reset(self):
        """Reset internal state at the start of each trial."""
        self.context_est = 0
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_state(self):
        return self.env.pos * 3 + self.context_est

    def update_context(self, obs):
        o_cue = obs % self.num_cues
        if o_cue in (1, 2):
            self.context_est = o_cue

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state, done):
        q_next = 0 if done else np.max(self.Q[next_state])
        self.Q[state, action] += self.alpha * (reward + self.gamma * q_next - self.Q[state, action])
