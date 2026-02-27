# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            newValues = util.Counter()
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    newValues[state] = 0
                    continue
                actions = self.mdp.getPossibleActions(state)
                if not actions:
                    newValues[state] = 0
                    continue
                qValues = []
                for action in actions:
                    q = 0
                    for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                        reward = self.mdp.getReward(state, action, nextState)
                        q += prob * (reward + self.discount * self.values[nextState])
                    qValues.append(q)
                newValues[state] = max(qValues)
            self.values = newValues


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        qValue = 0
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, nextState)
            qValue += prob * (reward + self.discount * self.values[nextState])
        return qValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None
        actions = self.mdp.getPossibleActions(state)
        if not actions:
            return None
        bestAction = None
        bestQValue = float('-inf')
        for action in actions:
            qValue = self.computeQValueFromValues(state, action)
            if qValue > bestQValue:
                bestQValue = qValue
                bestAction = action
        return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        for i in range(self.iterations):
            state = states[i % len(states)]
            if self.mdp.isTerminal(state):
                continue
            actions = self.mdp.getPossibleActions(state)
            if not actions:
                continue
            qValues = []
            for action in actions:
                q = 0
                for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    reward = self.mdp.getReward(state, action, nextState)
                    q += prob * (reward + self.discount * self.values[nextState])
                qValues.append(q)
            self.values[state] = max(qValues)

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # compute predecessors of all states.
        predecessors = {}
        states = self.mdp.getStates()
        for state in states:
            predecessors[state] = set()
        for state in states:
            for action in self.mdp.getPossibleActions(state):
                for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    if prob > 0:
                        predecessors[nextState].add(state)
        # initialize an empty priority queue
        pq = util.PriorityQueue()
        # For each non-terminal state, push it into the priority queue with priority equal to 
        # the absolute value of the difference between its current value and the highest Q-value 
        # across all possible actions from that state (this is 0 for terminal states) #
        for state in states:
            if self.mdp.isTerminal(state):
                continue
            actions = self.mdp.getPossibleActions(state)
            qValues = [self.computeQValueFromValues(state, action) for action in actions]
            maxQValue = max(qValues) if qValues else 0
            diff = abs(self.values[state] - maxQValue)
            pq.push(state, -diff)
        # For iteration in 0, 1, 2, ..., self.iterations - 1:
        for i in range(self.iterations):
            if pq.isEmpty():
                break
            state = pq.pop()
            if not self.mdp.isTerminal(state):
                actions = self.mdp.getPossibleActions(state)
                qValues = [self.computeQValueFromValues(state, action) for action in actions]
                self.values[state] = max(qValues) if qValues else 0
            for p in predecessors[state]:
                if self.mdp.isTerminal(p):
                    continue
                actions = self.mdp.getPossibleActions(p)
                qValues = [self.computeQValueFromValues(p, action) for action in actions]
                maxQValue = max(qValues) if qValues else 0
                diff = abs(self.values[p] - maxQValue)
                if diff > self.theta:
                    pq.update(p, -diff)