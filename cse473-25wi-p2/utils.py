import numpy as np

class FrozenLake:
    def __init__(self, grid_size=4, is_slippery=False, hole_prob=0.2, living_cost=-0.01, seed=42):
        """
        FrozenLake environment.

        Parameters:
        - grid_size: The size of the grid (grid_size x grid_size).
        - is_slippery: If True, movement will have stochastic effects.
        - hole_prob: Probability of a state being a hole (excludes start and goal).
        - living_cost: The cost (or reward) of agent transitioning between non-goal nodes.
        - seed: Random seed for reproducibility.
        """
        self.grid_size = grid_size
        self.nS = grid_size * grid_size # number of states
        self.nA = 4  # Actions: 0=Left, 1=Down, 2=Right, 3=Up
        self.is_slippery = is_slippery
        self.living_cost = living_cost
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Generate hole locations
        self.holes = self._generate_holes(hole_prob)

        # Build transition probabilities
        self.P = self._build_transition_probabilities()
        self.reset()

    def _generate_holes(self, hole_prob):
        """Randomly generate holes in the lake, excluding start (0) and goal (nS-1)."""
        num_holes = int(hole_prob * self.nS)  # Number of holes
        possible_holes = set(range(1, self.nS - 1))  # Exclude start and goal
        return set(self.rng.choice(list(possible_holes), size=num_holes, replace=False))

    def _build_transition_probabilities(self):
        """Builds the transition probabilities considering holes and slippery ice."""
        P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}

        for s in range(self.nS):
            for a in range(self.nA):
                possible_transitions = self._compute_transition(s, a)
                for prob, next_state, reward, done in possible_transitions:
                    P[s][a].append((prob, next_state, reward, done))
        return P

    def _compute_transition(self, s, a):
        """Computes the transitions for a given state and action."""
        if s in self.holes or s == self.nS - 1:  # Terminal states (holes or goal)
            return [(1.0, s, 0.0, True)]  # Stays in the same state if terminal

        directions = {
            0: (0, -1),  # Left
            1: (1, 0),   # Down
            2: (0, 1),   # Right
            3: (-1, 0)   # Up
        }

        row, col = divmod(s, self.grid_size)
        transitions = []

        if self.is_slippery:
            # Slippery: move in three possible directions with equal probability
            slip_directions = self._get_slip_directions(a)
            prob = 1 / len(slip_directions)  # Equal probability for each slip direction
        else:
            # Deterministic: move in intended direction only
            slip_directions = [a]
            prob = 1.0

        for move in slip_directions:
            dr, dc = directions[move]
            new_row, new_col = row + dr, col + dc

            # Check if new position is within grid bounds
            if 0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size:
                next_state = new_row * self.grid_size + new_col
            else:
                next_state = s  # Bounce back if hitting boundary

            reward = 1.0 if next_state == self.nS - 1 else self.living_cost
            done = next_state in self.holes or next_state == self.nS - 1
            transitions.append((prob, next_state, reward, done))

        return transitions

    def _get_slip_directions(self, a):
        """Returns the possible directions when slipping on ice."""
        slip_map = {
            0: [3, 0, 1],  # Left → {Up, Left, Down}
            1: [0, 1, 2],  # Down → {Left, Down, Right}
            2: [1, 2, 3],  # Right → {Down, Right, Up}
            3: [2, 3, 0]   # Up → {Right, Up, Left}
        }
        return slip_map[a]

    def reset(self):
        """Resets the environment to the start state."""
        self.state = 0
        return self.state

    def step(self, action):
        """Executes an action and returns (probability, next_state, reward, done)."""
        transitions = self.P[self.state][action]
        probs, _, _, _ = zip(*transitions)

        chosen_idx = self.rng.choice(len(transitions), p=probs)
        prob, next_state, reward, done = transitions[chosen_idx]

        self.state = next_state
        return prob, next_state, reward, done

    def render(self):
        """Prints the FrozenLake grid as ASCII art."""
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        for hole in self.holes:
            row, col = divmod(hole, self.grid_size)
            grid[row][col] = 'H'

        grid[0][0] = 'S'  # Start position
        grid[-1][-1] = 'G'  # Goal position

        print()
        for row in grid:
            print(" ".join(row))
