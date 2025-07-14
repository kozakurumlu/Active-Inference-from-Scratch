import numpy as np
from itertools import product

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

class ActiveInferenceAgent:
    def __init__(self, environment):
        self.model = environment
        self.q_s = self.model.D.copy()
        self.horizon = 3  # Planning depth

    def reset(self):
        self.q_s = np.zeros(self.model.num_states)
        start_state_0 = self.model._get_state_index(0, 0)
        start_state_1 = self.model._get_state_index(0, 1)
        self.q_s[[start_state_0, start_state_1]] = 0.5

    def infer_states(self, observation):
        log_likelihood = np.log(self.model.A[observation, :] + 1e-16)
        log_prior = np.log(self.q_s + 1e-16)
        self.q_s = softmax(log_likelihood + log_prior)

    def select_action(self):
        G_list = []
        policies = list(product(range(self.model.num_actions), repeat=self.horizon))
        w = 1.0  # Weight for epistemic value
        for pi in policies:
            q_s_tau = self.q_s.copy()
            G_pi = 0.0
            for t in range(self.horizon):
                a = pi[t]
                q_s_tau = self.model.B[:, :, a] @ q_s_tau
                q_o_tau = self.model.A @ q_s_tau
                pragmatic_value = q_o_tau @ np.log(self.model.C + 1e-16)
                H_O = -np.sum(q_o_tau * np.log(q_o_tau + 1e-16))
                expected_H_O_given_S = -np.sum(self.model.A * np.log(self.model.A + 1e-16), axis=0)
                epistemic_value = H_O - np.dot(q_s_tau, expected_H_O_given_S)
                G_pi += pragmatic_value + w * epistemic_value
            G_list.append(G_pi)
        policy_probs = softmax(np.array(G_list) * 10.0)
        chosen_idx = np.random.choice(len(policies), p=policy_probs)
        chosen_action = policies[chosen_idx][0]
        self.q_s = self.model.B[:, :, chosen_action] @ self.q_s
        return chosen_action

    def learn_from_experience(self, state_belief, obs, learning_rate=0.1):
        o_one_hot = np.zeros(self.model.num_obs)
        o_one_hot[obs] = 1.0
        update = learning_rate * np.outer(o_one_hot, state_belief)
        self.model.A += update
        self.model.A /= self.model.A.sum(axis=0)