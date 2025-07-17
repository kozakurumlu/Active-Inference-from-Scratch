import numpy as np

def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


class ActiveInferenceAgent:
    """Active Inference agent with simplified state uncertainty."""

    def __init__(self, env, learning_rate=1.0, epistemic_weight=1.0):
        self.env = env
        self.lr = learning_rate
        self.epistemic_weight = epistemic_weight

        # generative model: start with a weak prior close to the true mapping
        prior = env.build_true_A()
        self.dirichlet = prior * 10.0
        self.A = self.dirichlet / np.sum(self.dirichlet, axis=0, keepdims=True)

        # known deterministic transitions for position
        self.pos = 0
        self.q_c = np.array([0.5, 0.5])  # belief over context

        self.reset()

    def reset(self):
        """Initial state belief at start of trial."""
        self.pos = 0
        self.q_c[:] = 0.5
        self.last_action = None

    def infer_states(self, obs):
        """Update beliefs about hidden state using Bayes rule."""
        # update internal position belief from last action
        if self.last_action is not None:
            self.pos = self._next_pos(self.pos, self.last_action)
        # update context belief knowing current position exactly
        ll = np.array([
            self.A[obs, self.env._state_index(self.pos, 0)],
            self.A[obs, self.env._state_index(self.pos, 1)],
        ])
        self.q_c = self.q_c * ll
        self.q_c = self.q_c / np.sum(self.q_c)

    def learn(self, obs):
        """Dirichlet learning of observation model."""
        for ctx, q in enumerate(self.q_c):
            idx = self.env._state_index(self.pos, ctx)
            self.dirichlet[obs, idx] += self.lr * q
        self.A = self.dirichlet / np.sum(self.dirichlet, axis=0, keepdims=True)

    def _next_pos(self, pos, action):
        if pos == 0 and action == 0:
            return 1
        if pos == 1 and action == 1:
            return 2
        if pos == 1 and action == 2:
            return 3
        return pos

    def select_action(self):
        """Action that maximizes expected reward and information gain."""
        H_prior = -np.sum(self.q_c * np.log(self.q_c + 1e-16))
        values = []
        step_cost = -0.1
        for a in range(self.env.num_actions):
            next_pos = self._next_pos(self.pos, a)
            # expected reward including step cost
            reward = step_cost
            if next_pos == 2:
                reward += self.q_c[0] - self.q_c[1]
            elif next_pos == 3:
                reward += self.q_c[1] - self.q_c[0]
            # information gain
            H_post = H_prior
            if next_pos == 1:
                H_post = 0.0
                for o in range(self.env.num_obs):
                    p_o = 0.0
                    post = []
                    for ctx in range(2):
                        idx = self.env._state_index(next_pos, ctx)
                        p = self.q_c[ctx] * self.A[o, idx]
                        post.append(p)
                        p_o += p
                    if p_o > 0:
                        post = np.array(post) / p_o
                        H_post += p_o * (-np.sum(post * np.log(post + 1e-16)))
            info_gain = H_prior - H_post
            values.append(reward + self.epistemic_weight * info_gain)
        probs = softmax(np.array(values))
        action = int(np.random.choice(len(values), p=probs))
        self.last_action = action
        return action


