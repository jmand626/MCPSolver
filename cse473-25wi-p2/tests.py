import unittest
import numpy as np

from utils import FrozenLake
from solutions import value_iteration, policy_iteration, q_learning

class FrozenLakeSanityCheckTests(unittest.TestCase):
    def setUp(self):
        """Initialize a simple FrozenLake environment."""
        self.env = FrozenLake(grid_size=4, is_slippery=False, living_cost=-0.01, hole_prob=0.2)
        print("\n\nHere is the FrozenLake setup for these tests:")
        self.env.render()

    def test_value_iteration(self):
        """Sanity check for value_iteration function."""
        V = value_iteration(self.env)
        self.assertIsInstance(V, np.ndarray)
        self.assertEqual(V.shape, (self.env.nS,))
        self.assertTrue(np.all(V >= 0))  # Ensure non-negative values
        self.assertTrue(np.max(V) <= 1.0)  # Ensure values are within reasonable bounds

    def test_policy_iteration(self):
        """Sanity check for policy_iteration function."""
        policy = policy_iteration(self.env)
        self.assertIsInstance(policy, np.ndarray)
        self.assertEqual(policy.shape, (self.env.nS,))
        self.assertTrue(np.all(np.logical_and(policy >= 0, policy < self.env.nA)))  # Ensure valid actions
        self.assertTrue(np.any(policy >= 0))  # Ensure policy is non-trivial

    def test_q_learning(self):
        """Sanity check for q_learning function."""
        Q = q_learning(self.env, num_episodes=100)
        self.assertIsInstance(Q, np.ndarray)
        self.assertEqual(Q.shape, (self.env.nS, self.env.nA))
        self.assertTrue(np.max(Q) <= 1.0)  # Ensure values are within reasonable bounds

    def test_policy_iteration_q_learning_match(self):
        """Sanity check that the same optimal policy is found between two techniques."""
        # calculate optimal policy via PI
        pi_policy = policy_iteration(self.env)
        print("\n\nVisualizing policy iteration policy:")
        self._pretty_print_policy(pi_policy)

        # calculate optimal policy via q-learning
        self.env.reset()
        Q = q_learning(self.env, num_episodes=100000)
        Q_policy = np.argmax(Q, axis=1)
        print("\n\nVisualizing q-learning policy:")
        self._pretty_print_policy(Q_policy)

        # Count number of differing elements between policies.
        # This is proxy for similar policies, most of the time it's 0.
        # But we bound from above by 1 because q-learning is non-deterministic with epsilon-greedy.
        self.assertTrue(np.sum(pi_policy != Q_policy) <= 1)

    def _pretty_print_policy(self, policy: np.ndarray):
        grid = [['.' for _ in range(self.env.grid_size)] for _ in range(self.env.grid_size)]
        directions = {
            0: "L",  # Left
            1: "D",  # Down
            2: "R",  # Right
            3: "U"   # Up
        }

        for idx, policy_action in enumerate(policy):
            if idx not in self.env.holes and idx != self.env.nS - 1:
                row, col = divmod(idx, self.env.grid_size)
                grid[row][col] = directions.get(policy_action)
        print()
        for row in grid:
            print(" ".join(row))

if __name__ == '__main__':
    unittest.main()
