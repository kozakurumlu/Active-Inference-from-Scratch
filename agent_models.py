import numpy as np
from itertools import product


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


class ActiveInferenceAgent:
    """Active Inference agent for the T-maze."""

    def __init__(self, env, learning_rate=1.0, planning_horizon=4, epistemic_weight=1.0):
        self.env = env
        self.learning_rate = learning_rate
        self.horizon = planning_horizon
        self.epistemic_weight = epistemic_weight

        # Generative model components
        self.A = np.ones((env.num_obs, env.num_states)) / env.num_obs
        self.dirichlet = np.ones((env.num_obs, env.num_states))
        self.B = env.build_B()
        # Reward structure over states
        self.R = np.full(env.num_states, -0.1)
        left_correct = env._state_index(2, 0)
        right_correct = env._state_index(3, 1)
        left_wrong = env._state_index(2, 1)
        right_wrong = env._state_index(3, 0)
        self.R[left_correct] = 1.0
        self.R[right_correct] = 1.0
        self.R[left_wrong] = -1.0
        self.R[right_wrong] = -1.0

        self.reset()

    def reset(self):
        """Initial state belief at start of trial."""
        s0_left = self.env._state_index(0, 0)
        s0_right = self.env._state_index(0, 1)
        self.q_s = np.zeros(self.env.num_states)
        self.q_s[s0_left] = 0.5
        self.q_s[s0_right] = 0.5
        self.last_action = None

    def infer_states(self, obs):
        """Update beliefs about hidden state given observation."""
        prior = self.q_s
        if self.last_action is not None:
            prior = self.B[:, :, self.last_action] @ self.q_s
        ll = np.log(self.A[obs, :] + 1e-16)
        lp = np.log(prior + 1e-16)
        self.q_s = softmax(ll + lp)

    def learn(self, obs):
        """Dirichlet learning of observation model."""
        self.dirichlet[obs, :] += self.learning_rate * self.q_s
        self.A = self.dirichlet / np.sum(self.dirichlet, axis=0, keepdims=True)

    def select_action(self):
        """Sample an action by minimizing expected free energy."""
        policies = list(product(range(self.env.num_actions), repeat=self.horizon))
        values = []
        for pi in policies:
            qs = self.q_s.copy()
            v = 0.0
            for a in pi:
                qs = self.B[:, :, a] @ qs
                qo = self.A @ qs
                pragmatic = qs @ self.R
                # posterior over context
                qc0 = qs[[self.env._state_index(p, 0) for p in range(self.env.num_positions)]].sum()
                qc1 = qs[[self.env._state_index(p, 1) for p in range(self.env.num_positions)]].sum()
                qc = np.array([qc0, qc1])
                entropy_c = -np.sum(qc * np.log(qc + 1e-16))
                epistemic = -entropy_c
                v += pragmatic + self.epistemic_weight * epistemic
            values.append(v)
        probs = softmax(np.array(values) * 10.0)
        idx = np.random.choice(len(policies), p=probs)
        action = policies[idx][0]
        # state prediction after committing to action
        self.q_s = self.B[:, :, action] @ self.q_s
        self.last_action = action
        return action


