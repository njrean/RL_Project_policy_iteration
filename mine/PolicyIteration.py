from GridWorld import GridWorld
import numpy as np
import matplotlib.pyplot as plt

class PolicyIteration:
    def __init__(self, World, gamma, init_policy=None):
        self.world = World
        self.num_states = World.num_states
        self.num_actions = World.num_actions
        self.reward_function = World.reward_map
        # self.transition_model = transition_model
        self.gamma = gamma
        self.values = np.zeros(self.num_states)

        if init_policy is None:
            self.policy = np.random.choice(self.world.action_space, size=self.num_states)
        else:
            self.policy = init_policy

    def policy_evaluation(self):
        delta = 0
        for s in range(self.num_states):
            temp = self.values[s]
            a = self.policy[s]
            p = self.world.transition_prob() 
            self.values[s] = self.reward_function[s] + self.gamma * np.sum(p * self.values)
            delta = max(delta, abs(temp - self.values[s]))
        return delta

