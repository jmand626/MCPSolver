# Reinforcement Learning on FrozenLake

This project provides an introduction to fundamental **Reinforcement Learning (RL)** algorithms applied to the classic **FrozenLake** environment. It includes implementations of **Value Iteration**, **Policy Iteration**, and **Q-Learning**. 

## The FrozenLake Environment

FrozenLake is a simple gridworld environment. **The Frozen Lake environment is a square grid containing four types of tiles**: Start (S), Frozen (F), Hole (H), and Goal (G). The agent starts at S and must navigate to G by moving up, down, left, or right, without falling into any H holes. Stepping onto the goal yields a reward of +1, while all other moves give `living_cost` reward. Falling into a hole terminates the episode with 0 reward. 

**State Space:** The state is represented by the agent’s position on the grid (e.g. there are 16 states in a 4x4 grid). State 0 is the starting position and state 15 is the goal.
**Action Space:** There are 4 discrete actions: 0 = Left, 1 = Down, 2 = Right, 3 = Up.  

The environment’s dynamics are defined by transition probabilities. For a given state and action, the environment may move to a new state with a certain probability and provide a reward. (In the non-slippery setting, these transitions are deterministic.) The FrozenLake environment provides a handy dictionary `env.P` to access these dynamics. For example, `env.P[s][a]` returns a list of tuples `(prob, next_state, reward, done)` for taking action `a` in state `s`. `prob` is the transition probability itself, `next_state` is the successor `s'`, `reward` is the reward value associated with the transition, and `done` is a boolean value indicating if the state is terminal (True).

## Project Structure and Usage

The project contains the following files:
- **`solutions.py`** – Function stubs for students to implement (`value_iteration`, `policy_iteration`, `q_learning`).
- **`utils.py`** – Helper functions for the assignment, contains the `FrozenLake` definition itself.
- **`README.md`** – This file, explaining how to use the code.
- **`tests.py`** – Unit tests (using Python’s `unittest` framework) to sanity check your implementations.

### Requirements

- Python 3.10
- NumPy (`pip install numpy`)

### Running the Tests

To validate your implementations, run:
```bash
python -m unittest -v tests 
