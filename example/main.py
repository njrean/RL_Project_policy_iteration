from grid_world import GridWorld
from policy_iteration import PolicyIteration
import numpy as np
import os

# cwd = os.getcwd()
# print the current directory
# print("Current working directory is:", cwd)

problem = GridWorld('example/data/world00.csv', reward={0: -0.1, 1: 1.0, 2: -1.0, 3: np.NaN}, random_rate=0.2)
print(problem.num_rows)
policy = [1, 1, 3, 1, 0, 0, 2, 0, 1, 2, 1, 0]

solver = PolicyIteration(problem.reward_function, problem.transition_model, gamma=0.9, init_policy=policy)
# solver.train()

print(problem.transition_model.shape)

# problem.visualize_value_policy(policy=solver.policy, values=solver.values)
# problem.random_start_policy(policy=solver.policy, start_pos=(2, 0), n=1000)

#C:\Users\natch\Github\RL_Project_policy_iteration\example\data\world00.csv