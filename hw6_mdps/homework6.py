############################################################
# Imports
############################################################

############################################################
# CIS 5210: Homework 6
############################################################

student_name = ""

############################################################
# Section 1: Value Iteration
############################################################


class ValueIterationAgent:
    """Implement Value Iteration Agent using Bellman Equations."""

    def __init__(self, game, discount):
        """Store game object and discount, initialize state values to 0."""
        self.game = game
        self.discount = discount
        self.values = {state: 0 for state in game.states}

    def get_value(self, state):
        """Return value V*(s) correspond to state."""
        return self.values.get(state, 0)

    def get_q_value(self, state, action):
        """Return Q*(s,a) correspond to state and action.

        Q*(s,a) = sum_s' T(s,a,s') [R(s,a,s') + gamma * V*(s')]
        """
        return sum(
            prob * (self.game.get_reward(state, action, next_state)
                    + self.discount * self.get_value(next_state))
            for next_state, prob
            in self.game.get_transitions(state, action).items()
        )

    def get_best_policy(self, state):
        """Return policy pi*(s) correspond to state.

        pi*(s) = argmax_a Q*(s,a)
        """
        actions = self.game.get_actions(state)
        if not actions:
            return None
        return max(actions, key=lambda a: self.get_q_value(state, a))

    def iterate(self):
        """Run single batch value iteration using Bellman equation.

        V_{k+1}(s) = max_a Q*(s,a), computed from fixed V_k for all s,
        then stored as the new V*.
        """
        new_values = {
            state: max(
                self.get_q_value(state, action)
                for action in self.game.get_actions(state)
            )
            for state in self.game.states
        }
        self.values = new_values


############################################################
# Section 2: Policy Iteration
############################################################


class PolicyIterationAgent(ValueIterationAgent):
    """Implement Policy Iteration Agent.

    The only difference between policy iteration and value iteration is at
    their iteration method. However, if you need to implement helper
    functions or override ValueIterationAgent's methods, you can add them.
    """

    def iterate(self):
        """Run single policy iteration.

        Fix current policy, iterate state values V(s) until
        |V_{k+1}(s) - V_k(s)| < epsilon for all s, then update values.
        """
        epsilon = 1e-6
        policy = {s: self.get_best_policy(s) for s in self.game.states}
        while True:
            new_values = {
                s: self.get_q_value(s, policy[s])
                for s in self.game.states
            }
            delta = max(
                abs(new_values[s] - self.values.get(s, 0))
                for s in self.game.states
            )
            self.values = new_values
            if delta < epsilon:
                break


############################################################
# Section 3: Bridge Crossing Analysis
############################################################


def question_3():
    discount = 0.9
    noise = 0.0
    return discount, noise


############################################################
# Section 4: Policies
############################################################


def question_4a():
    # Close exit (+1), risk cliff: low discount (prefer close), no noise
    discount = 0.1
    noise = 0.0
    living_reward = 0.0
    return discount, noise, living_reward


def question_4b():
    # Close exit (+1), avoid cliff: low discount (prefer close), moderate noise
    discount = 0.3
    noise = 0.3
    living_reward = 0.0
    return discount, noise, living_reward


def question_4c():
    # Distant exit (+10), risk cliff: high discount (prefer distant), no noise
    discount = 0.9
    noise = 0.0
    living_reward = 0.0
    return discount, noise, living_reward


def question_4d():
    # Distant exit (+10), avoid cliff: high discount, some noise
    discount = 0.9
    noise = 0.2
    living_reward = 0.0
    return discount, noise, living_reward


def question_4e():
    # Avoid all exits and cliff: large living reward makes cycling best
    discount = 0.9
    noise = 0.0
    living_reward = 100.0
    return discount, noise, living_reward


############################################################
# Section 5: Feedback
############################################################


# Just an approximation is fine.
feedback_question_1 = """
4 hours
"""

feedback_question_2 = """
The batch vs. in-place distinction in value iteration tripped me up
for a while. The spec bolds it, but I did not fully appreciate the
difference until my intermediate values looked off compared to the
hint image. Once I realized I was updating V in place during the same
sweep, the fix was obvious, but it cost me more debugging time than
I expected.

Section 4 was harder than it looked. I could reason about discount
and noise individually but predicting how they interact on a specific
grid took more trial and error than I anticipated. The bridge question
was actually the cleaner one since you can only touch one parameter,
which forced me to think carefully about why noise alone causes the
agent to stay near the safe exit rather than risk crossing.
"""

feedback_question_3 = """
I liked that value iteration and policy iteration both converge to the
same result but get there in completely different ways. Watching that
in the GUI, where policy iteration reached the right policy in far
fewer iterations, made the efficiency argument from lecture feel
concrete rather than just theoretical.

If I could change anything, I would add a small worked example for
section 4 showing one completed case alongside the grid layout. It
took me a bit to realize that "risking the cliff" is about which row
the policy follows under noise-free execution, not whether the agent
randomly falls off mid-episode.
"""

############################################################
# Section 6: Acknowledgements & reflections
############################################################

# Did you collaborate with anyone or use generative AI?
acknowledgements = """
I used Claude to help me when I was stuck with this homework
not by asking for the direct answer but ask me questions to
prompt me to develop and write the answer on my own. Using an AI
tool sped up my thinking process but I practiced tracing
through edge cases to make sure I understood what each function does.

What helped me understand the material:
- Before looking at any code, I asked Claude to walk me through the
  difference between batch and online value iteration (why batch
  computes all V_{k+1} from a frozen V_k rather than updating in
  place), which is exactly what the "Important" note in the spec
  flags. Having that distinction clear made the iterate() logic
  straightforward to read and verify.
- For policy iteration, I asked it to explain what "policy evaluation"
  means mechanically (fix pi, run Bellman backup for that action only
  until convergence) before looking at the code. That helped me see
  why the inner while-loop converges faster than full value iteration.
- For sections 3 and 4, I asked Claude to walk through the Bellman
  values for specific states on the bridge and discount grids under
  different parameter choices, then cross-checked by running the
  converged agent computationally and inspecting each state's policy.

I read through every method line by line and ran the agent on all the
provided grids to confirm the results matched expectations before
submitting.
"""
