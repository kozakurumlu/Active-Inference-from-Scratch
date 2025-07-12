"""
Active Inference from Scratch: Extended T-Maze Problem with Learning

This script implements an advanced Active Inference (AcI) agent to solve an extended T-Maze task.
We build on the basic T-Maze by introducing:
1. **Larger State Space**: Expanded maze with 7 positions (0: start, 1: stem, 2: junction/cue, 3: left mid, 4: left end/rewardA, 5: right mid, 6: right end/rewardB).
   - This increases complexity, requiring more steps to reach the cue or rewards, amplifying epistemic exploration.
2. **Stochasticity and Noise**: Probabilistic transitions (0.8 success rate, 0.1 slip left/right, 0.1 stay) and lower observation accuracy (0.8) to model real-world uncertainty.
3. **Parameter Learning**: Use Dirichlet priors for likelihood A and transitions B. Update posteriors over trials by accumulating counts based on inferred states/observations/actions.
   - This enables cross-trial learning: Start with uncertain/inaccurate model, refine via free energy minimization over parameters (Bayesian updating of concentrations).

The goal is deep understanding: Concepts tied to Free Energy Principle (FEP) - agents minimize surprise via inference (belief updates), action (expected free energy minimization), and now learning (model adaptation) for self-organization and ergodicity in uncertain environments.

Structure:
1. **ExtendedTMaze class**: Environment and generative model, now with Dirichlet for A/B.
2. **ActiveInferenceAgent class**: Perception (variational inference), action (G minimization), and parameter updates.
3. **Simulation**: Run trials, plot learning curves (reward rate, VFE over time).
"""

import numpy as np
import matplotlib.pyplot as plt

# Helper: Softmax for probabilities
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

# --- 1. The Generative Model and Environment ---
class ExtendedTMaze:
    """
    Defines the extended T-Maze environment and agent's generative model.
    - Hidden states s: position (7) x context (2) = 14 states.
    - Observations o: position x cue (3: neutral, A, B) = 21 obs.
    - Actions: 4 (up=0, down=1, left=2, right=3).
    - Rewards: +1 at pos4 (ctx A) or pos6 (ctx B); episode ends at 4/6.
    - Cue: Informative only at pos2 (junction).
    - Stochastic transitions: 0.8 to intended, 0.1 stay, 0.1 slip (random adjacent).
    - Parameter learning: Dirichlet priors for A (obs likelihood) and B (transitions).
      - Concentrations updated post-trial via accumulated q(s), o, a, q(s').
    """
    def __init__(self, context):
        # Dimensions of the environment and agent's model
        self.num_positions = 7  # 0:start,1:stem,2:junction,3:left mid,4:left end,5:right mid,6:right end
        self.num_contexts = 2   # 0:A (reward left/4), 1:B (reward right/6)
        self.num_states = self.num_positions * self.num_contexts  # 14 hidden states
        self.num_obs_pos = self.num_positions
        self.num_obs_cue = 3  # 0:neutral,1:A-cue,2:B-cue
        self.num_obs = self.num_obs_pos * self.num_obs_cue  # 21 possible observations
        self.num_actions = 4  # 0:up,1:down,2:left,3:right

        # True environment context (hidden from agent)
        self.true_context = context
        self.true_state = self.true_context  # Start at pos0 + ctx

        # --- Dirichlet Priors for Learning ---
        # Initial Dirichlet concentrations for A: p(o|s) ~ Dir(alpha_A)
        # Start with weak prior (1.0 uniform) for learning
        self.alpha_A = np.ones((self.num_obs, self.num_states))  # Will update with experience
        self.A = self._dir_to_prob(self.alpha_A)  # Initial p(o|s)

        # Dirichlet for B: p(s'|s,a) ~ Dir(alpha_B per s,a)
        self.alpha_B = np.ones((self.num_states, self.num_states, self.num_actions))
        self.B = self._dir_to_prob(self.alpha_B)  # Initial p(s'|s,a); axis=0 sums to 1 per s,a

        # Goal prior: High for reward states (no learning here)
        self.goal_prior = np.zeros(self.num_states)
        self.goal_prior[4 * 2 + 0] = 10.0  # pos4 + A (left end, context A)
        self.goal_prior[6 * 2 + 1] = 10.0  # pos6 + B (right end, context B)
        self.goal_prior = softmax(self.goal_prior)  # Softmax for normalized preferences

        # Initial state prior D: Uniform (maximal uncertainty at start)
        self.D = np.ones(self.num_states) / self.num_states

    def _dir_to_prob(self, alpha):
        """Convert Dirichlet concentrations to probabilities (normalize columns)."""
        prob = alpha / np.sum(alpha, axis=0, keepdims=True)
        return prob

    def _get_position(self, s): return s // self.num_contexts
    def _get_context(self, s): return s % self.num_contexts

    def step(self, action):
        """
        Simulate a step in the true environment:
        - Applies the action to the true state (with stochasticity)
        - Generates a noisy observation for the agent
        - Returns: observation index, reward, done flag
        """
        # --- Transition Logic ---
        current_pos = self._get_position(self.true_state)
        next_pos = current_pos  # Default: stay in place
        # Determine intended move based on action
        if action == 0:  # up
            if current_pos in [0,1]: next_pos += 1  # 0->1,1->2
        elif action == 1:  # down
            if current_pos in [1,2,3,5]: next_pos -= 1  # 1->0,2->1,3->2,5->2
            elif current_pos in [4,6]: next_pos = current_pos  # Ends stay
        elif action == 2:  # left
            if current_pos == 2: next_pos = 3
            elif current_pos == 3: next_pos = 4
        elif action == 3:  # right
            if current_pos == 2: next_pos = 5
            elif current_pos == 5: next_pos = 6

        # --- Stochasticity: Intended, Stay, or Slip ---
        rand = np.random.rand()
        if rand < 0.8:
            pass  # 80%: intended move
        elif rand < 0.9:
            next_pos = current_pos  # 10%: stay in place
        else:
            # 10%: slip to adjacent (±1 if possible)
            adj = [max(0, current_pos-1), min(6, current_pos+1)]
            if len(set(adj)) > 1:  # Avoid duplicate if at edge
                next_pos = np.random.choice(adj)
            else:
                next_pos = adj[0]

        # --- Update true state (context never changes) ---
        next_state = next_pos * self.num_contexts + self.true_context
        self.true_state = next_state

        # --- Generate observation (noisy, context-specific cue at junction) ---
        accuracy = 0.8  # True environment's observation accuracy
        noise = (1 - accuracy) / (self.num_obs - 1)
        true_A_col = noise * np.ones(self.num_obs)
        pos = next_pos
        cue = 0 if pos != 2 else self.true_context + 1  # Only at junction is cue informative
        expected_o = pos * self.num_obs_cue + cue
        true_A_col[expected_o] = accuracy
        # Sample observation based on true likelihood
        o_idx = np.random.choice(self.num_obs, p=true_A_col / np.sum(true_A_col))

        # --- Reward and Episode End ---
        done = pos in [4, 6]  # End if at left or right end
        reward = 1 if (pos == 4 and self.true_context == 0) or (pos == 6 and self.true_context == 1) else -1 if done else 0

        return o_idx, reward, done

# --- 2. The Active Inference Agent ---
class ActiveInferenceAgent:
    """
    AcI agent with perception, action, and parameter learning.
    - Perception: Update q(s|o) via approximate variational inference (iterative).
    - Action: Minimize G(π) = - (epistemic + pragmatic) for horizon=1.
    - Learning: Post-step, accumulate to alpha_A and alpha_B using q(s), o, a, q(s').
      - For A: Add q(s) to alpha_A[o, s] (observed likelihood).
      - For B: Add q(s) to alpha_B[s', s, a] (transition counts).
      - Regenerate A/B from updated alphas after each trial.
    Ties to FEP: Learning minimizes model evidence lower bound, adapting to reduce long-term surprise.
    """
    def __init__(self, model):
        self.model = model
        self.q_s = self.model.D.copy()  # Current beliefs (posterior over states)

    def reset(self):
        # Reset beliefs to prior at start of each trial
        self.q_s = self.model.D.copy()

    def perceive(self, o_idx):
        """
        Perception (state inference):
        - Given an observation, update belief over hidden states (q_s)
        - Uses iterative variational inference to minimize variational free energy (VFE)
        - q_s is updated to reflect how likely each state is given the new observation
        """
        o_onehot = np.zeros(self.model.num_obs)
        o_onehot[o_idx] = 1.0
        prior_q = self.q_s.copy()
        num_iters, lr = 10, 1.0  # Number of inference steps, learning rate
        for _ in range(num_iters):
            lnA_o = np.log(self.model.A.T @ o_onehot + 1e-16)
            grad = np.log(self.q_s + 1e-16) - np.log(prior_q + 1e-16) + 1 - lnA_o
            self.q_s -= lr * grad * self.q_s
            self.q_s = softmax(self.q_s)
        # Track VFE for analysis
        lnA_o = np.log(self.model.A.T @ o_onehot + 1e-16)
        evidence = self.q_s @ lnA_o
        kl = self.q_s @ (np.log(self.q_s + 1e-16) - np.log(prior_q + 1e-16))
        self.vfe = kl - evidence  # Variational free energy (lower is better)

    def act(self):
        """
        Action selection:
        - For each possible action, predict the next state distribution (q_s')
        - Compute expected free energy (G) for each action, balancing:
            * Epistemic value (expected information gain)
            * Pragmatic value (expected reward/goal alignment)
        - Select the action with minimum G (most valuable)
        - Update q_s to predicted q_s' for next step
        """
        G = np.zeros(self.model.num_actions)
        for a in range(self.model.num_actions):
            # Predict next state distribution after action a
            q_sp = self.model.B[:, :, a] @ self.q_s
            # --- Epistemic Value: Expected information gain ---
            q_op = self.model.A @ q_sp  # Predicted observation distribution
            H_q_op = -np.sum(q_op * np.log(q_op + 1e-16))  # Entropy of predicted obs
            H_p_o_s = -np.sum(self.model.A * np.log(self.model.A + 1e-16), axis=0)  # Entropy per state
            expected_H = q_sp @ H_p_o_s  # Expected entropy
            epistemic = H_q_op - expected_H  # Info gain
            # --- Pragmatic Value: Goal alignment ---
            pragmatic = q_sp @ np.log(self.model.goal_prior + 1e-16)
            # --- Expected Free Energy ---
            G[a] = - (epistemic + pragmatic)
        # Choose action with minimum expected free energy
        action = np.argmin(G)
        # Update belief to predicted next state (for next perception cycle)
        self.q_s = self.model.B[:, :, action] @ self.q_s
        return action

    def update_parameters(self, prev_q_s, o_idx, a, next_q_s, learning_rate=0.1):
        """
        Parameter learning (model adaptation):
        - After each step, update Dirichlet concentration parameters for A and B
        - For A: Add inferred prev_q_s to alpha_A[o, s] (i.e., how likely was each state to have produced the observed o)
        - For B: Add prev_q_s[s] * next_q_s[s'] to alpha_B[s', s, a] (i.e., how likely was the transition s->s' under action a)
        - Regenerate A and B as normalized probabilities from updated alphas
        """
        # Update A: Add inferred prev_q_s to observed o row
        self.model.alpha_A[o_idx, :] += learning_rate * prev_q_s
        # Update B: Add prev_q_s[s] * next_q_s[s'] to alpha_B[s', s, a]
        for s in range(self.model.num_states):
            for sp in range(self.model.num_states):
                self.model.alpha_B[sp, s, a] += learning_rate * prev_q_s[s] * next_q_s[sp]
        # Regenerate A and B from updated Dirichlet parameters
        self.model.A = self.model._dir_to_prob(self.model.alpha_A)
        self.model.B = self.model._dir_to_prob(self.model.alpha_B)

# --- 3. Simulation ---
def run_simulation(num_trials=100, max_steps=30):
    """
    Simulate multiple trials to demonstrate learning:
    - Each trial: agent starts with prior, interacts with environment, updates beliefs and model
    - Context alternates for diversity
    - After each step, agent updates model parameters (A, B) based on experience
    - Tracks cumulative reward rate and average VFE to show learning progress
    """
    rewards, vfes = [], []  # Track reward and free energy per trial
    cum_reward_rate = []    # Running average of reward rate
    # Persistent model across trials for learning
    context = 0  # Start with A, alternate
    model = ExtendedTMaze(context)  # Shared model instance
    for trial in range(num_trials):
        context = trial % 2  # Alternate context each trial
        model.true_context = context
        model.true_state = model.true_context  # Reset to start position
        agent = ActiveInferenceAgent(model)
        agent.reset()
        total_reward, trial_vfe, done, t = 0, 0.0, False, 0
        # Initial observation (null action -1 for start)
        o_idx, _, _ = model.step(-1 if t == 0 else agent.a)  # Get initial observation
        while not done and t < max_steps:
            prev_q_s = agent.q_s.copy()  # Store belief before perception/action
            agent.perceive(o_idx)        # Update belief based on observation
            trial_vfe += agent.vfe       # Accumulate VFE for analysis
            a = agent.act()              # Select action based on expected free energy
            o_idx, reward, done = model.step(a)  # Environment responds
            next_q_s = agent.q_s.copy()  # Belief after action
            agent.update_parameters(prev_q_s, o_idx, a, next_q_s)  # Learn from experience
            total_reward += reward
            t += 1
        # Track performance metrics
        rewards.append(1 if total_reward > 0 else 0)  # 1 if correct reward, else 0
        vfes.append(trial_vfe / max(1, t))            # Average VFE per trial
        cum_reward_rate.append(np.mean(rewards))      # Running average
        if (trial + 1) % 10 == 0:
            print(f"Trial {trial+1}: Reward {rewards[-1]}, Avg VFE {vfes[-1]:.2f}")

    # --- Plotting Learning Curves ---
    fig, ax1 = plt.subplots()
    ax1.plot(cum_reward_rate, 'b-', label='Cum Reward Rate')
    ax1.set_ylabel('Reward Rate')
    ax1.set_ylim(0, 1)
    ax2 = ax1.twinx()
    ax2.plot(vfes, 'r--', label='Avg VFE')
    ax2.set_ylabel('VFE')
    plt.title('AcI Learning in Extended T-Maze')
    plt.xlabel('Trial')
    fig.legend(loc='upper left')
    plt.show()

if __name__ == '__main__':
    run_simulation()