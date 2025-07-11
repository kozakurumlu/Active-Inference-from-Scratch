"""
Active Inference from Scratch: T-Maze Problem

This script implements a simple Active Inference agent to solve the T-maze task.
The goal is to provide a clear, step-by-step guide to the core concepts of
Active Inference, using only NumPy for numerical operations.

This tutorial is structured into three main parts:
1.  **The Generative Model (TMaze class):** This is the agent's internal model of the
    world. It defines how the agent believes states, observations, and actions
    are related.
2.  **The Active Inference Agent (ActiveInferenceAgent class):** This contains the
    core logic for perception and action, driven by the principle of minimizing
    free energy.
3.  **The Simulation (run_simulation function):** This runs the experiment and
    visualizes the agent's learning and performance.
"""

import numpy as np
import matplotlib.pyplot as plt

# --- 1. The Generative Model and Environment ---
# The heart of an Active Inference agent is its "generative model." This is not just
# a component of the agent; it IS the agent's reality. The agent uses this model
# to predict and explain its sensations. We define both the environment and the
# agent's model of it in this class.

class TMaze:
    """
    Defines the T-Maze environment and the agent's generative model.
    
    The world is modeled as a Partially Observable Markov Decision Process (POMDP),
    which consists of:
    - Hidden States (s): The true state of the world, which the agent cannot directly see.
      In our case, this is the agent's (position, context) pair.
    - Observations (o): The sensory data the agent receives. This is a noisy version
      of the true state.
    - Actions (a): The actions the agent can take to influence the world.
    """
    def __init__(self, context):
        # --- State and Observation Space Dimensions ---
        # We first define the size of our world.
        self.num_positions = 4  # 0: start, 1: junction, 2: left arm, 3: right arm
        self.num_contexts = 2   # 0: Context A (reward is left), 1: Context B (reward is right)
        
        # A "state" is a combination of a position and a context.
        self.num_states = self.num_positions * self.num_contexts
        
        # An "observation" is a combination of a perceived position and a perceived cue.
        self.num_obs_pos = self.num_positions
        self.num_obs_cue = 3  # 0: neutral cue, 1: A-cue, 2: B-cue
        self.num_obs = self.num_obs_pos * self.num_obs_cue
        
        self.num_actions = 4  # 0: up, 1: down, 2: left, 3: right

        # --- The "True" Environment ---
        # This part of the class represents the actual world, which the agent does not
        # have direct access to. We use it to generate observations for the agent.
        self.true_context = context
        # The agent always starts at position 0 in the given context.
        self.true_state = 0 * self.num_contexts + self.true_context

        # --- The Agent's Generative Model (Internal Beliefs) ---
        # These matrices define the agent's beliefs about how the world works.
        
        # A: Likelihood Matrix p(o|s)
        # This matrix answers: "If the true state is s, what am I likely to observe?"
        # It maps hidden states to observations. Each column is a probability distribution.
        self.A = self._create_likelihood_matrix()

        # B: Transition Matrix p(s'|s, a)
        # This matrix answers: "If I am in state s and I take action a, what state s' will I be in next?"
        # It defines the agent's understanding of the consequences of its actions.
        self.B = self._create_transition_matrix()
        
        # Goal Prior: p*(s)
        # This vector represents the agent's preferences or goals, expressed as a
        # probability distribution over states. It answers: "Which states do I want to be in?"
        self.goal_prior = self._create_goal_prior()

        # D: Initial State Prior p(s)
        # This vector represents the agent's belief about where it starts.
        # We initialize it to be uniform (the agent is maximally uncertain).
        self.D = np.ones(self.num_states) / self.num_states

    # Helper functions to easily switch between state index and position/context
    def _get_position(self, s): return s // self.num_contexts
    def _get_context(self, s): return s % self.num_contexts

    def _create_likelihood_matrix(self):
        """Builds the A matrix, defining the agent's belief about its senses."""
        A = np.zeros((self.num_obs, self.num_states))
        accuracy = 0.95  # How much the agent trusts its senses.
        
        for s in range(self.num_states):
            pos = self._get_position(s)
            ctx = self._get_context(s)
            
            # The agent believes the context cue is only visible at the junction (pos 1).
            expected_cue = ctx + 1 if pos == 1 else 0
            expected_o = pos * self.num_obs_cue + expected_cue
            
            # The agent believes its observation will be accurate with high probability.
            noise = (1 - accuracy) / (self.num_obs - 1)
            A[:, s] = noise
            A[expected_o, s] = accuracy
        
        # Normalize to ensure columns are valid probability distributions.
        return A / A.sum(axis=0, keepdims=True)

    def _create_transition_matrix(self):
        """Builds the B matrix, defining the agent's belief about action outcomes."""
        B = np.zeros((self.num_states, self.num_states, self.num_actions))
        
        for s in range(self.num_states):
            pos = self._get_position(s)
            ctx = self._get_context(s)
            
            for a in range(self.num_actions):
                next_pos = pos # By default, actions that are not allowed result in staying put.
                if pos == 0 and a == 0:   # At start, 'up' moves to junction
                    next_pos = 1
                elif pos == 1 and a == 2: # At junction, 'left' moves to left arm
                    next_pos = 2
                elif pos == 1 and a == 3: # At junction, 'right' moves to right arm
                    next_pos = 3
                
                # The agent believes the context does not change.
                next_s = next_pos * self.num_contexts + ctx
                # The agent believes its actions have deterministic outcomes.
                B[next_s, s, a] = 1.0
        return B

    def _create_goal_prior(self):
        """Builds the goal_prior vector, defining the agent's preferences."""
        goal_prior = np.zeros(self.num_states)
        # The agent prefers to be in the state that corresponds to the reward location.
        # State for (pos 2, context A): s = 2 * 2 + 0 = 4
        # State for (pos 3, context B): s = 3 * 2 + 1 = 7
        goal_prior[4] = 1.0 
        goal_prior[7] = 1.0
        # We don't normalize here; the magnitude reflects the strength of the preference.
        return goal_prior

    def step(self, action):
        """Simulates one step in the true environment."""
        current_pos = self._get_position(self.true_state)
        
        # Update the true position based on the action.
        next_pos = current_pos
        if current_pos == 0 and action == 0: next_pos = 1
        elif current_pos == 1 and action == 2: next_pos = 2
        elif current_pos == 1 and action == 3: next_pos = 3
        
        # The true context remains the same.
        self.true_state = next_pos * self.num_contexts + self.true_context
        
        # Generate the observation for the agent.
        cue = self.true_context + 1 if next_pos == 1 else 0
        obs_idx = next_pos * self.num_obs_cue + cue
        
        # Check if the episode has ended.
        done = next_pos in [2, 3]
        
        # Provide a reward if the agent reached the correct goal.
        reward = 1 if (self.true_context == 0 and next_pos == 2) or \
                       (self.true_context == 1 and next_pos == 3) else 0
            
        return obs_idx, reward, done

# --- 2. The Active Inference Agent ---

def softmax(x):
    """A numerically stable softmax function to normalize beliefs."""
    b = np.max(x)
    y = np.exp(x - b)
    return y / y.sum()

class ActiveInferenceAgent:
    """
    This class implements the agent's perception and action cycles.
    """
    def __init__(self, model):
        self.model = model
        # The agent's belief about its current state, initialized with the prior D.
        self.q_s = self.model.D.copy()

    def reset(self):
        """Resets the agent's beliefs to the initial prior for a new trial."""
        self.q_s = self.model.D.copy()

    def perceive(self, o_idx):
        """
        State Estimation (Perception).
        The agent updates its belief about the current state based on the new observation.
        This is a form of Bayesian inference, where:
        Posterior ∝ Likelihood × Prior
        q(s)      ∝ p(o|s)   × p(s)
        We use logarithms to prevent numerical underflow (multiplying small probabilities).
        """
        log_likelihood = np.log(self.model.A[o_idx, :] + 1e-16)
        log_prior = np.log(self.q_s + 1e-16)
        
        # The new belief is the normalized product of likelihood and prior.
        self.q_s = softmax(log_likelihood + log_prior)

    def act(self):
        """
        Action Selection.
        The agent chooses the action that it expects to minimize the Expected Free Energy (G).
        G is a quantity that balances two competing drives:
        
        1. Pragmatic Value: The drive to be in preferred states.
        2. Epistemic Value: The drive to gather information and reduce uncertainty.
        
        The agent selects the action with the lowest G.
        """
        G = np.zeros(self.model.num_actions)
        
        for a in range(self.model.num_actions):
            # For each possible action, predict the future state distribution.
            q_s_prime = self.model.B[:, :, a] @ self.q_s
            
            # --- Calculate Pragmatic Value ---
            # This measures how much the predicted future states align with the agent's goals.
            # It is the dot product of the predicted state distribution and the log of the goal prior.
            pragmatic_value = np.sum(q_s_prime * np.log(self.model.goal_prior + 1e-16))

            # --- Calculate Epistemic Value ---
            # This measures how much information the agent expects to gain from an action.
            # It is the expected reduction in uncertainty about the world.
            # An action has high epistemic value if it leads to unambiguous observations.
            q_o_prime = self.model.A @ q_s_prime
            H_q_o_prime = -np.sum(q_o_prime * np.log(q_o_prime + 1e-16))
            
            p_o_given_s = self.model.A
            H_p_o_given_s = -np.sum(p_o_given_s * np.log(p_o_given_s + 1e-16), axis=0)
            expected_H = np.sum(q_s_prime * H_p_o_given_s)
            
            epistemic_value = H_q_o_prime - expected_H
            
            # G is the negative sum of pragmatic and epistemic values.
            # The agent wants to maximize these values, so it minimizes their negative.
            G[a] = -(pragmatic_value + epistemic_value)
            
        # Select the action with the minimum Expected Free Energy.
        action = np.argmin(G)
        
        # After selecting an action, the agent updates its belief about its state
        # to be the state it expects to be in after taking that action. This forms the
        # prior for the next perception cycle.
        self.q_s = self.model.B[:, :, action] @ self.q_s
        
        return action

# --- 3. Simulation ---

def run_simulation(num_trials=50):
    """Runs the T-maze simulation and plots the agent's performance."""
    all_rewards = []
    
    # Run the simulation for both possible contexts.
    for context in [0, 1]:
        print(f"--- Running for Context {('A' if context == 0 else 'B')} ---")
        env = TMaze(context=context)
        agent = ActiveInferenceAgent(model=env)
        
        context_rewards = []
        for trial in range(num_trials):
            # Reset the environment and the agent's beliefs for each new trial.
            env.true_state = 0 * env.num_contexts + env.true_context
            agent.reset()
            
            done = False
            total_reward = 0
            t = 0
            
            # Get the initial observation before the first action.
            obs, _, _ = env.step(-1) # Use a null action to get the starting observation.

            # Each trial consists of a few timesteps.
            while not done and t < 10:
                # 1. The agent perceives its environment.
                agent.perceive(obs)
                
                # 2. The agent decides on an action.
                action = agent.act()
                
                # 3. The environment responds to the action.
                obs, reward, done = env.step(action)
                
                total_reward += reward
                t += 1
            
            context_rewards.append(total_reward)
            if (trial + 1) % 10 == 0:
                print(f"Trial {trial+1}/{num_trials} | Reward: {total_reward}")
        
        all_rewards.append(context_rewards)

    # --- Plotting Results ---
    plt.figure(figsize=(12, 6))
    plt.plot(all_rewards[0], 'o', label='Context A (Reward Left)', alpha=0.5, markersize=4)
    plt.plot(all_rewards[1], 'x', label='Context B (Reward Right)', alpha=0.5, markersize=5)
    
    # Calculate and plot a moving average to show the learning trend.
    window = 10
    avg_a = np.convolve(all_rewards[0], np.ones(window)/window, mode='valid')
    avg_b = np.convolve(all_rewards[1], np.ones(window)/window, mode='valid')
    plt.plot(np.arange(window-1, num_trials), avg_a, label='Trend A', color='blue', linewidth=2)
    plt.plot(np.arange(window-1, num_trials), avg_b, label='Trend B', color='red', linewidth=2)

    plt.title('Active Inference Agent Performance on T-Maze', fontsize=16)
    plt.xlabel('Trial', fontsize=12)
    plt.ylabel('Reward (1 = Correct, 0 = Incorrect)', fontsize=12)
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    run_simulation()
