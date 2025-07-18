# Active Inference from Scratch

This project contains a minimal implementation of active inference and reinforcement learning agents in a noisy T-maze environment, using only NumPy and no external libraries for these agents. It is intended as a small demonstration and learning opportunity, showing how active inference can handle perceptual noise with less degradation in performance compared to a simple Q-learning agent.

## Overview

The repository includes four main Python files:

- `t_maze_env.py` – defines a basic T-maze where the agent must navigate from a start position to the correct goal. Observations are noisy and the environment can generate larger maps for future experiments.
- `aci_agent.py` – active inference (AcI) agent that maintains beliefs about context using a generative model and updates its observation model via Dirichlet learning.
- `rl_agent.py` – baseline Q-learning agent with an epsilon-greedy policy.
- `simulation.py` – runs a series of trials for each agent across different noise levels and reports overall accuracy.

## Results

When sweeping noise levels from `0.1` to `0.7` in `simulation.py`, the average drop in accuracy for the AcI agent is consistently smaller than for the Q-learning agent. For example, with 200 trials per noise level in a standard T-maze, the average accuracy drop from `0.1` noise to higher noise levels was around **9.7%** for the AcI agent versus **11.4%** for Q-learning:

```
AI {0.1: 0.84, 0.2: 0.795, 0.3: 0.74, 0.4: 0.74}
RL {0.1: 0.88, 0.835, 0.78, 0.725}
Drops 9.7 11.4
```

Although the Q-learning agent often achieves slightly higher raw accuracy at low noise, its performance decreases more sharply as observations become less reliable.

## Limitations and Future Work

The current map is intentionally small to keep the example easy to understand. Because there are only a few states, the reinforcement learning agent can simply memorise state–action pairs after enough trials. To truly highlight the benefits of active inference, the environment should be extended with a larger maze or additional uncertainty (for example, the `ExtendedTMazeEnv` class already provides a stochastic version with more positions). A bigger map would prevent the Q-learning agent from memorising the optimal path so quickly and better illustrate how the active inference agent's generative model scales with complexity.

## Running Experiments

Install the dependencies (mainly `matplotlib` and `tqdm`) and run:

```bash
python simulation.py
```

This will generate a bar plot comparing the accuracy of both agents at increasing noise levels and print a summary of the average accuracy drop. 
