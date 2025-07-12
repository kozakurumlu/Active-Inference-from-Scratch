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

    def act(self, trial_num, depth=3, sample_actions=1):  # Increased depth to 3 with full pruning (sample_actions=1)
        def recursive_g(q_current, d):
            if d == 0:
                return 0, 0  # No future epi/prag
            G_sub = np.zeros(self.model.num_actions)
            for a in range(self.model.num_actions):
                q_sp = self.model.B[:, :, a] @ q_current  # Predict next state
                q_op = self.model.A @ q_sp  # Predict obs
                # Local epistemic: How much info this action gives immediately
                H_q_op = -np.sum(q_op * np.log(q_op + 1e-16))  # Overall ambiguity
                H_p_o_s = -np.sum(self.model.A * np.log(self.model.A + 1e-16), axis=0)  # Per-state ambiguity
                expected_H = q_sp @ H_p_o_s
                epistemic = H_q_op - expected_H
                # Local pragmatic: How close to goals
                pragmatic = q_sp @ np.log(self.model.goal_prior + 1e-16)
                # Recurse for future, add to sub G for pruning
                future_ep, future_prag = recursive_g(q_sp, d-1)
                G_sub[a] = - (epistemic + pragmatic + future_ep + future_prag)  # Cumulative negative (min G = max value)
            # Prune: Select top sample_actions by lowest G_sub (best futures)
            top_as = np.argsort(G_sub)[:sample_actions]
            avg_ep, avg_prag = 0, 0
            for a in top_as:
                q_sp = self.model.B[:, :, a] @ q_current
                ep, pr = recursive_g(q_sp, d-1)  # Recurse only on promising
                avg_ep += ep / sample_actions
                avg_prag += pr / sample_actions
            return avg_ep, avg_prag

        G = np.zeros(self.model.num_actions)
        for a in range(self.model.num_actions):
            q_sp = self.model.B[:, :, a] @ self.q_s
            # Compute local + future
            q_op = self.model.A @ q_sp  # Predict obs
            H_q_op = -np.sum(q_op * np.log(q_op + 1e-16))  # Overall ambiguity
            H_p_o_s = -np.sum(self.model.A * np.log(self.model.A + 1e-16), axis=0)  # Per-state ambiguity
            expected_H = q_sp @ H_p_o_s
            local_ep = H_q_op - expected_H
            local_prag = q_sp @ np.log(self.model.goal_prior + 1e-16)
            future_ep, future_prag = recursive_g(q_sp, depth-1)
            G[a] = - (local_ep + local_prag + future_ep + future_prag)
        # Hybrid temp strategy: Start high (1.0) for exploration, decay to 0.3 for exploitation
        temp = 1.0 * (1 - trial_num / 500) + 0.3  # Decay over 500 trials
        p_a = softmax(-G / temp)
        action = np.random.choice(self.model.num_actions, p=p_a)
        self.q_s = self.model.B[:, :, action] @ self.q_s  # Update with selected action
        return action

    def update_parameters(self, prev_q_s, o_idx, a, next_q_s, learning_rate=0.1):
        self.model.alpha_A[o_idx, :] += learning_rate * prev_q_s
        for s in range(self.model.num_states):
            for sp in range(self.model.num_states):
                self.model.alpha_B[sp, s, a] += learning_rate * prev_q_s[s] * next_q_s[sp]
        self.model.A = self.model._dir_to_prob(self.model.alpha_A)
        self.model.B = self.model._dir_to_prob(self.model.alpha_B)