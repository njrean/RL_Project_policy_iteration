from GridWorld import GridWorld
from PolicyIteration import PolicyIteration

reward={0: 0, 1: -1, 2: -100, 3: 5}  # 0=>normal block, 1=>start, 2=>water, 3=>destination

world = GridWorld('policy_iteration/world.csv', reward=reward)
solver = PolicyIteration(world, gamma=0.1)

solver.policy_iteration_solve()
solver.result_visualize()
solver.state_value_visualize()

print('number of iterations:',len(solver.history_policy))

for i, policy in enumerate(solver.history_policy):
    print('iterataion: {}'.format(i))
    world.visualize_path(policy)