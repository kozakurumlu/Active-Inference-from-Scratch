"""
AcI agent for grid env.
- Perception: Variational inference on q(s).
- Action: G minimization.
- Learning: Dirichlet updates.
"""

import numpy as np
from environment import softmax

class ActiveInferenceAgent:
    def __init__(self, model):
        self.model = model
        self.q_s = self.model.D.copy()

    def reset(self):
        self.q_s = self.model.D.copy()

    def perceive(self, o_idx):
        o_onehot = np.zeros(self.model.num_obs)
        o_onehot[o_idx] = 1.0
        prior_q = self.q_s.copy()
        num_iters, lr = 10, 1.0
        for _ in range(num_iters):
            lnA_o = np.log(self.model.A.T @ o_onehot + 1e-16)
            grad = np.log(self.q_s + 1e-16) - np.log(prior_q + 1e-16) + 1 - lnA_o
            self.q_s -= lr * grad * self.q_s
            self.q_s = softmax(self.q_s)
        lnA_o = np.log(self.model.A.T @ o_onehot + 1e-16)
        evidence = self.q_s @ lnA_o
        kl = self.q_s @ (np.log(self.q_s + 1e-16) - np.log(prior_q + 1e-16))
        self.vfe = kl - evidence

    def act(self):
        G = np.zeros(self.model.num_actions)
        for a in range(self.model.num_actions):
            q_sp = self.model.B[:, :, a] @ self.q_s
            q_op = self.model.A @ q_sp
            H_q_op = -np.sum(q_op * np.log(q_op + 1e-16))
            H_p_o_s = -np.sum(self.model.A * np.log(self.model.A + 1e-16), axis=0)
            expected_H = q_sp @ H_p_o_s
            epistemic = H_q_op - expected_H
            pragmatic = q_sp @ np.log(self.model.goal_prior + 1e-16)
            G[a] = - (epistemic + pragmatic)
        action = np.argmin(G)
        self.q_s = self.model.B[:, :, action] @ self.q_s
        return action

    def update_parameters(self, prev_q_s, o_idx, a, next_q_s, learning_rate=0.1):
        self.model.alpha_A[o_idx, :] += learning_rate * prev_q_s
        for s in range(self.model.num_states):
            for sp in range(self.model.num_states):
                self.model.alpha_B[sp, s, a] += learning_rate * prev_q_s[s] * next_q_s[sp]
        self.model.A = self.model._dir_to_prob(self.model.alpha_A)
        self.model.B = self.model._dir_to_prob(self.model.alpha_B)