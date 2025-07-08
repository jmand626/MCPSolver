from utils import FrozenLake
import numpy as np
import random

def value_iteration(env: FrozenLake, gamma: float = 1.0, theta: float = 1e-8) -> np.ndarray:
    """
    Compute the optimal state-value function and policy for a given environment using Value Iteration.

    Parameters:
        env: The FrozenLake environment. It should have `env.P` (transition probabilities),
             `env.nS` for number of states, and `env.nA` for number of actions.
             See `utils.py` for the full definition.
        gamma (float): Discount factor (0 < gamma <= 1) for future rewards.
        theta (float): A small threshold for determining convergence (stop when value change < theta).

    Returns:
        V: Optimal value function, mapping each state to its value. This must be a NumPy array of size = number of states (env.nS).

    Hint: You can access transition probabilities and rewards via `env.P`. For example, env.P[s][a] returns a list of (prob, next_state, reward, done) tuples for state s and action a.
    """
    # We start off with the default value of the algo, as described in the first slide that details the algo. Note that
    # with these two slides, 11 and 12 from lecture mdp03, I will generally use the names from the Slide 11 version
    V = np.zeros(env.nS)
    # So zero out or function in the beginning. This numpy function will return the array type we want

    # I originally thought I should loop for a LIMITED amount with a for loop, but then I learned how to actually
    # read and saw the theta value above
    while True:
        # As we can in the slides, we need to track the change of values, so..
        delta = 0

        # So, now we recurse over each neighboring state
        for s in range(env.nS):
            oldVal = V[s]
            # To track delta changes

            # Now, we to continuously calculate the value we can get from each state, and track the maximum change that
            # we see when we do this. Therefore, we loop over all possible actions to states
            maxVal = float('-inf')

            for a in range(env.nA):
                value = 0
                for prob, next_state, reward, done in env.P[s][a]:
                    value += prob * (reward + gamma * V[next_state] * (not done))
                    # We should also include "not done" to account for the rewards that we get when we dont
                    # reach a terminal state, which obviously is very important

                # Remember that V(S) maximizes Q(S, a)
                if value > maxVal:
                    maxVal = value

            V[s] = maxVal
            # The whole point of maxVal and delta
            delta = max(delta, abs(oldVal - V[s]))

        if delta < theta:
            break
            # We have reached convergence!!!

    return V




def policy_iteration(env: FrozenLake, gamma: float = 1.0, theta: float = 1e-2) -> np.ndarray:
    """
    Compute the optimal policy for a given environment using Policy Iteration.

    Parameters:
        env: The FrozenLake environment.
        gamma (float): Discount factor for future rewards.
        theta (float): Convergence threshold for policy evaluation step (value function accuracy).

    Returns:
        policy: Optimal policy (This must be a NumPy array of size = number of states (env.nS).).
    """
    # We have seen this across all of these functions and across ed, so this should not be surprising, but what is new
    # is that we initialize BOTH V and pi here
    V = np.zeros(env.nS)
    P = np.zeros(env.nS)

    while True:
        stability = True

        # This outer loop goes over policy iteration and evaluation one after the other. The pseudocode does not make this
        # clear, but clear see at the end of step 3 that if we do not have a stable policy, we should go back to step 2
        while True:
            # Therefore, this inner loop is the evaluation which is so similar to value_iteration!!!!!!!!!!
            delta = 0

            # So, now we recurse over each neighboring state
            for s in range(env.nS):
                oldVal = V[s]
                # To track delta changes

                # Note that there is absolutely no reason to track the max value that we get!!!!!!!!!

                value = 0

                # The only thing that we change is that we dont go over all actions a, because we already have a policy
                # a that gives us the action
                for prob, next_state, reward, done in env.P[s][P[s]]:
                    value = prob * (reward + gamma * V[next_state] * (not done))
                    # We should also include "not done" to account for the rewards that we get when we dont
                    # reach a terminal state, which obviously is very important

                V[s] = value
                # The whole point of maxVal and delta
                delta = max(delta, abs(oldVal - V[s]))

            if delta < theta:
                break
            # We have reached convergence!!!

        # But remember that convergence is related to the convergence of VALUES, and this is the convergence of POLICY
        for s in range(env.nS):
            old = P[s]
            new = P[s]
            # Remember that policy is an array, not some kind of number, so its P[s], not P
            # Remember, an summation over the states means a for loop over them.
            argmax = float('-inf')
            val = 0
            for a in range(env.nA):
                val = 0
                for probability, next_state, reward, done in env.P[s][a]:
                    val += probability * (reward + gamma * V[next_state] * (not done))
                    # Same update rules from value iteration

                if val > argmax:
                    argmax = val

                    # Remember, the "policy" is just a dumb name for a rulebook that points where to go from each
                    # state, and a is definitely a way to go for one state in our policy, since its an action
                    new = a

            P[s] = new
            # new found us the best strategy for this state compared to the old strategy. Therefore we set it to new
            # in the rulebook


            # Now check if the policy even changed at all, because the previous code didnt care about if old != new
            if old != new:
                stability = False


        # Now we just check for stability to get out of this eternal hellish loop
        if stability:
            break


    # Freedom feelsStrongMan
    return P

def q_learning(env: FrozenLake, num_episodes: int, gamma: float = 1.0, alpha: float = 0.1, epsilon: float = 0.1) -> np.ndarray:
    """
    Train a Q-learning agent on the given environment.

    Parameters:
        env: The FrozenLake environment.
        num_episodes (int): Number of episodes to run the training for.
        gamma (float): Discount factor.
        alpha (float): Learning rate for Q-value updates.
        epsilon (float): Epsilon for the epsilon-greedy policy (probability of choosing a random action for exploration).

    Returns:
        Q: The learned Q-table, as a NumPy array with shape [env.nS x env.nA].
           Q[s][a] is the estimated value of taking action a in state s under the learned policy.

    Note: This algorithm does not require a model of the environment (it learns from experience),
          and uses an epsilon-greedy strategy to balance exploration and exploitation during training.
    """
    # Although this definitely is not easier than value_iteration, its definitely simpler than policy iteration, since
    # the pseudocode is just so much simpler and shorter
    Q = np.zeros((env.nS, env.nA))
    # "Initialize Q(s, a) for all s in S, a in A(s). Note that, crucially, we want to have a 2d matrix, and since the
    # zeroes needs a parameter that determines its dimension, so we need to send in a tuple.

    for _ in range(num_episodes):
    # Remember that we go over episodes/epochs before we even loop through our states/actions/etc, so we initialize
    # for each of that

        done = False
        # Boolean to break out of epoch loop when needed

        S = env.reset()

        while not done:
            # What is epsilon greedy? Its a mini-algo/procedure used in q-learning where you roll a die, and with high
            # probability you continue following your policy, but with a small amount of probability, you push the algo
            # to go explore something else for a change. Its like when you are 10 years old and your mom pushes you to go
            # talk with the other kids your age :|
            if np.random.random() >= epsilon:
                A = np.argmax(Q[S])
                # continue with our policy Q
            else:
                # Head into a random state. There is no guarantee that this random roll does not just push you into where
                # your policy would already go, but thats ok, this should happen many times
                A = np.random.randint(env.nA)


            prob, next_state, reward, done = env.step(A)
            # Actually take the step and get our results


            if not done:
                next_val = np.max(Q[next_state])
            else:
                next_val = 0

            # Now for our update rule!
            Q[S, A] = Q[S, A] + alpha * (reward + gamma * next_val - Q[S, A])
            # Remember the second piece after the first plus sign is basically the 'error' of the update!

            S = next_state

    return Q