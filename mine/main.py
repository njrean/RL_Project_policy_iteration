from GridWorld import GridWorld
from PolicyIteration import PolicyIteration
import numpy as np

reward={0: -0.4, 1: -0.4, 2: -5.0, 3: 5.0}  # 0=>normal block, 1=>start, 2=>water, 3=>destination

world = GridWorld('mine/world.csv', reward=reward)
solver = PolicyIteration(world, gamma=0.8)

print(solver.policy)
# world.visualize_path()
