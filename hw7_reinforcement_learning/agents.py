############################################################
# Imports
############################################################

# Include your imports here, if any are used.
import random

############################################################
# CIS 5210: Homework 7
############################################################

student_name = ""

############################################################
# Section 1: Q-Learning
# &
# Section 2: Epsilon Greedy
############################################################


class QLearningAgent:
    """Implement Q Reinforcement Learning Agent using Q-table."""

    def __init__(self, game, discount, learning_rate, explore_prob):
        """Store any needed parameters into the agent object.
        Initialize Q-table.
        """
        self.game = game
        self.discount = discount
        self.learning_rate = learning_rate
        self.explore_prob = explore_prob
        self.q_values = {}

    def get_q_value(self, state, action):
        """Retrieve Q-value from Q-table.
        For an never seen (s,a) pair, the Q-value is by default 0.
        """
        return self.q_values.get((state, action), 0)

    def get_value(self, state):
        """Compute state value from Q-values using Bellman Equation.
        V(s) = max_a Q(s,a)
        """
        actions = self.game.get_actions(state)
        if not actions:
            return 0
        return max(self.get_q_value(state, action) for action in actions)

    def get_best_policy(self, state):
        """Compute the best action for the state using Policy Extraction.
        π(s) = argmax_a Q(s,a)

        If there are ties, return a random one for better performance.
        Hint: use random.choice().
        """
        actions = self.game.get_actions(state)
        if not actions:
            return None
        best_value = self.get_value(state)
        best_actions = [
            action for action in actions
            if self.get_q_value(state, action) == best_value
        ]
        return random.choice(best_actions)

    def update(self, state, action, next_state, reward):
        """Update Q-values using running average.
        Q(s,a) = (1 - α) Q(s,a) + α (R + γ V(s'))
        Where α is the learning rate, and γ is the discount.

        Note: You should not call this function in your code.
        """
        old_q = self.get_q_value(state, action)
        sample = reward + self.discount * self.get_value(next_state)
        self.q_values[(state, action)] = (
            (1 - self.learning_rate) * old_q + self.learning_rate * sample
        )

    # 2. Epsilon Greedy
    def get_action(self, state):
        """Compute the action to take for the agent, incorporating exploration.
        That is, with probability ε, act randomly.
        Otherwise, act according to the best policy.

        Hint: use random.random() < ε to check if exploration is needed.
        """
        actions = self.game.get_actions(state)
        if not actions:
            return None
        if random.random() < self.explore_prob:
            return random.choice(list(actions))
        return self.get_best_policy(state)


############################################################
# Section 3: Bridge Crossing Revisited
############################################################


def question3():
    return 'NOT POSSIBLE'
    # If not possible, return 'NOT POSSIBLE'


############################################################
# Section 5: Approximate Q-Learning
############################################################


class ApproximateQAgent(QLearningAgent):
    """Implement Approximate Q Learning Agent using weights."""

    def __init__(self, *args, extractor):
        """Initialize parameters and store the feature extractor.
        Initialize weights table."""

        super().__init__(*args)
        self.extractor = extractor
        self.weights = {}

    def get_weight(self, feature):
        """Get weight of a feature.
        Never seen feature should have a weight of 0.
        """
        return self.weights.get(feature, 0)

    def get_q_value(self, state, action):
        """Compute Q value as the dot product of features and weights.
        Q(s,a) = w_1 * f_1(s,a) + w_2 * f_2(s,a) + ... + w_n * f_n(s,a)
        """
        features = self.extractor(state, action)
        return sum(
            self.get_weight(feature) * value
            for feature, value in features.items()
        )

    def update(self, state, action, next_state, reward):
        """Update weights using least-squares approximation.
        Δ = R + γ V(s') - Q(s,a)
        Then update weights: w_i = w_i + α * Δ * f_i(s, a)
        """
        delta = (
            reward + self.discount * self.get_value(next_state)
            - self.get_q_value(state, action)
        )
        features = self.extractor(state, action)
        for feature, value in features.items():
            self.weights[feature] = (
                self.get_weight(feature) + self.learning_rate * delta * value
            )


############################################################
# Section 6: Feedback
############################################################


# Just an approximation is fine.
feedback_question_1 = """
4 hours
"""

feedback_question_2 = """
Getting the abstraction boundary right was the trickiest part. The
spec warns that get_value and get_best_policy must only ever call
get_q_value rather than touching the Q-table directly, and I initially
underestimated why that mattered until I plugged in ApproximateQAgent
and watched get_action and get_value automatically start using the
feature-weighted Q-values with zero changes to their own code. That
was a nice "aha" moment once it clicked.

Bridge Crossing Revisited was the most conceptually demanding part.
It's tempting to think turning epsilon all the way down just removes
noise from an otherwise fine agent, but tracing through what happens
when the west exit gets discovered before the agent ever tries the
full eastward crossing made it clear why greedy-only exploration can
permanently lock onto the closer, smaller reward. Reasoning about the
tradeoff between epsilon being too low (never escapes the near exit)
and too high (risks falling off the cliff partway through the long
crossing) took a few passes with pen and paper before I trusted my
answer.
"""

feedback_question_3 = """
I liked that this assignment made the shift from "planning" to
"learning" concrete. In homework6 the agent computed an entire policy
before touching the environment; here the same Bellman-style update
happens incrementally from single (s, a, s', r) transitions, and
watching the Q-values in the Gridworld GUI slowly converge toward the
value-iteration numbers from last time was satisfying.

Approximate Q-learning was my favorite part. Seeing the exact same
weight-update code let Pacman generalize across an entire mediumGrid
after only 50 training games, when the tabular agent barely learned
anything useful even after thousands of games, made the point about
feature generalization land much harder than reading about it would
have.

If I could change one thing, I'd ask for a short note on why unseen
actions defaulting to Q=0 can look "optimistic" and affect
get_best_policy/get_action, similar to the explicit hint given for
the Pacman section. I ran into that exact issue while debugging and
would have caught it sooner with an earlier pointer.
"""

############################################################
# Section 7: Acknowledgements & reflections
############################################################

# Did you collaborate with anyone or use generative AI?
acknowledgements = """
I used Claude to help me when I was stuck with this homework
not by asking for the direct answer but ask me questions to
prompt me to develop and write the answer on my own. Using an AI
tool sped up my thinking process but I practiced tracing
through edge cases to make sure I understood what each function does.

After help with the implementation, I read through every method
against the formulas in the spec (the running-average Q-update, the
epsilon-greedy rule, and the weight update Δ = R + γV(s') - Q(s,a))
to confirm each line matched, and I ran the Gridworld GUI under
manual control to check that four episodes along the optimal path
produced Q-values matching the reference screenshot before moving on
to Pacman.


"""
