import gym
import numpy as np
import time

desc=["SFFG", "FFFF", "FFFF", "HFFF"]
env = gym.make("FrozenLake-v1",desc=desc, render_mode="human", is_slippery=True)
env.reset()
env.render()

print('Initial state of the system')
numberOfIterations=30

for i in range(numberOfIterations):
    randomAction= env.action_space.sample()
    returnValue=env.step(randomAction)
    print(returnValue)
    env.render()
    # print('Iteration: {} and action {}'.format(i+1,randomAction))
    time.sleep(0.5)
    if returnValue[2]:
        break

env.close()  