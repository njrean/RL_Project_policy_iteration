import numpy as np
import gym
import time

class GridWorld:
    def __init__(self, filename, reward, action_space=[0,1,2,3]):
        file = open(filename)
        self.map = np.array(
            [list(map(float, s.strip().split(","))) for s in file.readlines()]
        )
        file.close()

        self.grid_rows = self.map.shape[0]
        self.grid_cols = self.map.shape[1]
        self.num_states = self.grid_rows * self.grid_cols
        self.action_space = action_space
        self.num_actions = len(action_space)
        self.reward = reward                                #reward dictionary
        self.reward_map = self.generate_reward_map()
        self.desc = self.generate_desc()
        self.env = gym.make("FrozenLake-v1",desc=self.desc, render_mode="human", is_slippery=True)

    def state_to_pos(self, state):                          #pos is (x,y) coordinate 
        x = state % self.grid_cols
        y = state // self.grid_cols
        return x, y

    def pos_to_state(self, pos):                            #state is s0 = 0, s1 = 1, s2 = 2, ...
        state = (pos[1]*self.grid_cols)+ pos[0]    
        return state
    
    def generate_reward_map(self, ):
        reward_map = np.zeros((self.grid_rows, self.grid_cols))
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                reward = self.reward[self.map[r, c]]
                reward_map[r][c] = reward
        #reward map give a reward of each state by using [y][x] 
        return reward_map
    
    def generate_desc(self):
        desc_dict = {0:'F', 1:'S', 2:'H', 3:'G'}
        desc = []
        for r in range(self.grid_rows):
            s = ''
            for o in self.map[r]:
                s += desc_dict[o]
            desc.append(s)
        return desc

    def transition_prob(self):
        #need to modify in the future now every action in each state is equal to 0.33 
        prob = 1/3
        return prob
    
    def visualize_path(self):
        ########
        self.env.reset()
        self.env.render()
        time.sleep(4)

