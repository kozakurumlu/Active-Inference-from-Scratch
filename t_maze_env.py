import numpy as np

class TMazeEnv:
    """Simple T-maze environment with observational noise."""
    def __init__(self, noise_level=0.1, max_steps=10):
        self.noise_level = noise_level
        self.max_steps = max_steps

        # Positions: 0-start, 1-junction, 2-left goal, 3-right goal
        self.num_positions = 4
        self.num_contexts = 2  # 0: left correct, 1: right correct

        self.num_states = self.num_positions * self.num_contexts
        self.num_cues = 3  # 0-neutral, 1-left, 2-right
        self.num_obs = self.num_positions * self.num_cues

        self.num_actions = 3  # 0-Up, 1-Left, 2-Right

        self.reset()

    def _state_index(self, pos, ctx):
        return pos * self.num_contexts + ctx

    def reset(self):
        """Start a new trial with random hidden context."""
        self.context = np.random.randint(self.num_contexts)
        self.pos = 0
        self.step_count = 0
        return self._get_observation()

    def step(self, action):
        """Update position given action and return (obs, reward, done)."""
        if self.pos == 0 and action == 0:
            self.pos = 1
        elif self.pos == 1:
            if action == 1:
                self.pos = 2
            elif action == 2:
                self.pos = 3
        # otherwise action has no effect

        self.step_count += 1
        done = False
        reward = -0.1  # step penalty

        if self.pos in (2, 3):
            done = True
            if (self.context == 0 and self.pos == 2) or (
                self.context == 1 and self.pos == 3
            ):
                reward = 1.0
            else:
                reward = -1.0
        elif self.step_count >= self.max_steps:
            done = True

        obs = self._get_observation()
        return obs, reward, done

    def _get_observation(self):
        """Return noisy observation of current position and cue."""
        if self.pos == 1:
            cue = self.context + 1  # informative cue
        else:
            cue = 0
        true_obs = self.pos * self.num_cues + cue
        if np.random.rand() > self.noise_level:
            return true_obs
        # pick a random wrong observation
        choices = list(range(self.num_obs))
        choices.remove(true_obs)
        return np.random.choice(choices)

    def build_true_A(self):
        """Return the true observation likelihood matrix used by the agent."""
        A = np.zeros((self.num_obs, self.num_states))
        for pos in range(self.num_positions):
            for ctx in range(self.num_contexts):
                s = self._state_index(pos, ctx)
                if pos == 1:
                    cue = ctx + 1
                else:
                    cue = 0
                true_obs = pos * self.num_cues + cue
                for o in range(self.num_obs):
                    if o == true_obs:
                        A[o, s] = 1 - self.noise_level
                    else:
                        A[o, s] = self.noise_level / (self.num_obs - 1)
        return A

    def build_B(self):
        """Return deterministic transition matrix used by the agent."""
        B = np.zeros((self.num_states, self.num_states, self.num_actions))
        for pos in range(self.num_positions):
            for ctx in range(self.num_contexts):
                s = self._state_index(pos, ctx)
                # action 0: Up
                next_pos = pos
                if pos == 0:
                    next_pos = 1
                B[self._state_index(next_pos, ctx), s, 0] = 1.0

                # action 1: Left
                next_pos = pos
                if pos == 1:
                    next_pos = 2
                B[self._state_index(next_pos, ctx), s, 1] = 1.0

                # action 2: Right
                next_pos = pos
                if pos == 1:
                    next_pos = 3
                B[self._state_index(next_pos, ctx), s, 2] = 1.0
        return B

