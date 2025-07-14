import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

class GridWorld:
    def __init__(self, grid_size=3, noise_level=0.1):
        self.grid_size = grid_size
        self.num_positions = grid_size * grid_size
        self.num_contexts = 2  # 0: Goal is Top-Right, 1: Goal is Bottom-Left
        self.num_states = self.num_positions * self.num_contexts
        self.num_actions = 4  # 0: Up, 1: Down, 2: Left, 3: Right
        self.num_cues = 3
        self.num_obs = self.num_positions * self.num_cues
        self.cue_pos = self.grid_size // 2 * self.grid_size + self.grid_size // 2  # Center
        self.goal_pos = {
            0: self.grid_size - 1,  # Top-right
            1: self.num_positions - self.grid_size  # Bottom-left
        }
        self.true_context = 0
        self.agent_pos = 0
        self.true_state_idx = self._get_state_index(self.agent_pos, self.true_context)
        self.noise_level = noise_level
        self._init_generative_model()

    def _get_state_index(self, pos, context):
        return pos * self.num_contexts + context

    def _get_pos_and_context(self, state_idx):
        pos = state_idx // self.num_contexts
        context = state_idx % self.num_contexts
        return pos, context

    def _init_generative_model(self):
        # Fixed true likelihood for generating observations with parameterized noise
        self.true_A = np.zeros((self.num_obs, self.num_states))
        for s in range(self.num_states):
            pos, context = self._get_pos_and_context(s)
            for o_pos in range(self.num_positions):
                for o_cue in range(self.num_cues):
                    o_idx = o_pos * self.num_cues + o_cue
                    pos_is_correct = (o_pos == pos)
                    cue_is_correct = (o_cue == 0)  # Default neutral
                    if pos == self.cue_pos or pos == self.goal_pos[context]:
                        cue_is_correct = (o_cue == context + 1)
                    if pos_is_correct and cue_is_correct:
                        self.true_A[o_idx, s] = 1 - self.noise_level
                    else:
                        self.true_A[o_idx, s] = self.noise_level / (self.num_obs - 1)
        self.true_A /= np.sum(self.true_A, axis=0)

        # Agent's B, C, D (shared)
        self.B = np.zeros((self.num_states, self.num_states, self.num_actions))
        for s in range(self.num_states):
            pos, context = self._get_pos_and_context(s)
            row, col = divmod(pos, self.grid_size)
            for a in range(self.num_actions):
                next_row, next_col = row, col
                if a == 0 and row > 0: next_row -= 1
                elif a == 1 and row < self.grid_size - 1: next_row += 1
                elif a == 2 and col > 0: next_col -= 1
                elif a == 3 and col < self.grid_size - 1: next_col += 1
                next_pos = next_row * self.grid_size + next_col
                next_s_idx = self._get_state_index(next_pos, context)
                self.B[next_s_idx, s, a] = 1.0

        self.C = np.zeros(self.num_obs)
        self.C[self.goal_pos[0] * self.num_cues + 1] = 1.0
        self.C[self.goal_pos[1] * self.num_cues + 2] = 1.0
        self.C = softmax(self.C)

        self.D = np.ones(self.num_states) / self.num_states

        # Agent's learned A starts with a weak prior (20% true mapping + 80% uniform) to break symmetry and enable learning
        uniform = np.ones((self.num_obs, self.num_states)) / self.num_obs
        self.initial_A = 0.2 * self.true_A + 0.8 * uniform
        self.initial_A /= self.initial_A.sum(axis=0)
        self.A = self.initial_A.copy()

    def reset(self):
        self.true_context = np.random.randint(self.num_contexts)
        self.agent_pos = 0
        self.true_state_idx = self._get_state_index(self.agent_pos, self.true_context)
        return self.get_observation()

    def step(self, action):
        row, col = divmod(self.agent_pos, self.grid_size)
        next_row, next_col = row, col
        if action == 0 and row > 0: next_row -= 1
        elif action == 1 and row < self.grid_size - 1: next_row += 1
        elif action == 2 and col > 0: next_col -= 1
        elif action == 3 and col < self.grid_size - 1: next_col += 1
        self.agent_pos = next_row * self.grid_size + next_col
        self.true_state_idx = self._get_state_index(self.agent_pos, self.true_context)
        reward = -0.1
        done = False
        correct_goal = self.goal_pos[self.true_context]
        if self.agent_pos == correct_goal:
            reward = 1.0
            done = True
        elif self.agent_pos in self.goal_pos.values():
            reward = -1.0
            done = True
        return self.get_observation(), reward, done

    def get_observation(self):
        true_obs_dist = self.true_A[:, self.true_state_idx]
        sum_dist = np.sum(true_obs_dist)
        if sum_dist > 0:
            true_obs_dist = true_obs_dist / sum_dist
        else:
            true_obs_dist = np.ones(self.num_obs) / self.num_obs
        return np.random.choice(self.num_obs, p=true_obs_dist)