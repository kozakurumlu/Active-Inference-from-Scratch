"""
Grid world environment and generative model for AcI.
- Larger 2D grid (10x10 default) with obstacles (walls).
- States: pos x contexts = 200 (flattened).
- Obs: pos x cue = 300 (cue reveals context at center).
- Transitions: Stochastic (0.8 intended, 0.1 stay, 0.1 slip), blocked by obstacles.
- Learning: Dirichlet for A (likelihood) and B (transitions).
"""

import numpy as np

class GridWorldEnv:
    def __init__(self, context, grid_size=10):
        # Dimensions - Scalable; larger grid amplifies need for epistemic efficiency.
        self.grid_size = grid_size
        self.num_positions = self.grid_size ** 2  # e.g., 100
        self.num_contexts = 2
        self.num_states = self.num_positions * self.num_contexts  # e.g., 200
        self.num_obs_pos = self.num_positions
        self.num_obs_cue = 3  # neutral, A, B
        self.num_obs = self.num_obs_pos * self.num_obs_cue  # e.g., 300
        self.num_actions = 4  # up, down, left, right

        # Obstacles: List of blocked pos (e.g., a cross-shaped wall; customize).
        self.obstacles = [i for i in range(45, 55)]  # Horizontal wall in middle row

        # Cue position: Center for info foraging challenge.
        self.cue_pos = (self.grid_size // 2) * self.grid_size + (self.grid_size // 2)  # e.g., 55

        # Goals: Context-dependent, far apart for sparse rewards.
        self.goal_a_pos = self.grid_size - 1  # Top-right-ish, adjust
        self.goal_b_pos = (self.grid_size - 1) * self.grid_size  # Bottom-left-ish

        # True env
        self.true_context = context
        self.true_state = self.true_context  # Start at pos 0 + ctx

        # Dirichlet priors - Weak uniform for learning from scratch.
        self.alpha_A = np.ones((self.num_obs, self.num_states))
        self.A = self._dir_to_prob(self.alpha_A)
        self.alpha_B = np.ones((self.num_states, self.num_states, self.num_actions))
        self.B = self._dir_to_prob(self.alpha_B)

        # Goal prior - Biased to reward states.
        self.goal_prior = np.zeros(self.num_states)
        self.goal_prior[self.goal_a_pos * 2 + 0] = 10.0  # Goal A + ctx A
        self.goal_prior[self.goal_b_pos * 2 + 1] = 10.0  # Goal B + ctx B
        self.goal_prior = softmax(self.goal_prior)  # Defined below

        # Initial D
        self.D = np.ones(self.num_states) / self.num_states

    def _dir_to_prob(self, alpha):
        return alpha / np.sum(alpha, axis=0, keepdims=True)

    def _get_xy(self, pos): return divmod(pos, self.grid_size)  # row, col
    def _get_pos(self, row, col): return row * self.grid_size + col
    def _get_position(self, s): return s // self.num_contexts
    def _get_context(self, s): return s % self.num_contexts

    def step(self, action):
        current_pos = self._get_position(self.true_state)
        row, col = self._get_xy(current_pos)
        next_row, next_col = row, col
        if action == 0 and row > 0: next_row -= 1  # up
        elif action == 1 and row < self.grid_size - 1: next_row += 1  # down
        elif action == 2 and col > 0: next_col -= 1  # left
        elif action == 3 and col < self.grid_size - 1: next_col += 1  # right

        next_pos = self._get_pos(next_row, next_col)
        if next_pos in self.obstacles: next_pos = current_pos  # Blocked

        # Stochastic - Adds uncertainty; agent learns via B updates.
        rand = np.random.rand()
        if rand >= 0.8:
            if rand < 0.9:
                next_pos = current_pos
            else:
                adj = []
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                        np_pos = self._get_pos(nr, nc)
                        if np_pos not in self.obstacles:
                            adj.append(np_pos)
                if adj:
                    next_pos = np.random.choice(adj)

        next_state = next_pos * self.num_contexts + self.true_context
        self.true_state = next_state

        # Obs with noise - 0.8 accuracy.
        accuracy = 0.8
        noise = (1 - accuracy) / (self.num_obs - 1)
        true_A_col = noise * np.ones(self.num_obs)
        pos = next_pos
        cue = 0 if pos != self.cue_pos else self.true_context + 1
        expected_o = pos * self.num_obs_cue + cue
        true_A_col[expected_o] = accuracy
        o_idx = np.random.choice(self.num_obs, p=true_A_col / true_A_col.sum())

        # Reward/done
        done = pos in [self.goal_a_pos, self.goal_b_pos]
        reward = 1 if (pos == self.goal_a_pos and self.true_context == 0) or (pos == self.goal_b_pos and self.true_context == 1) else -1 if done else 0

        return o_idx, reward, done

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)