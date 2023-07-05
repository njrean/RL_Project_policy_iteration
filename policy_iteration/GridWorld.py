import numpy as np
import gym
import time

class GridWorld:
    def __init__(self, filename, reward, action_space=[0,1,2,3], action_dict=None):
        file = open(filename)
        self.map = np.array([list(map(float, s.strip().split(","))) for s in file.readlines()])
        file.close()

        self.grid_rows = self.map.shape[0]
        self.grid_cols = self.map.shape[1]
        self.num_states = self.grid_rows * self.grid_cols
        self.action_space = action_space
        self.num_actions = len(action_space)
        self.reward = reward                                #reward dictionary
        self.reward_map = self.generate_reward_map()
        self.desc = self.generate_desc()
        self.transition_prob = self.generate_transition_prob()
        self.env = gym.make("FrozenLake-v1",desc=self.desc, render_mode="human", is_slippery=False)
        self.env_color = {0:'#dcdedc', 1:'#67e670', 2:'#f25769', 3:'#fac825'}

        if action_dict == None:
            self.action_dict = {0:'←', 1:'↓', 2:'→', 3:'↑'}
        else:
            self.action_dict = action_dict

    def state_to_pos(self, state):                          #pos is (x,y) coordinate 
        x = state % self.grid_cols
        y = state // self.grid_cols
        return x, y

    def pos_to_state(self, x, y):                            #state is s0 = 0, s1 = 1, s2 = 2, ...
        if x < 0 or y < 0 or x > self.grid_cols-1 or y > self.grid_rows-1:
            state = -1
        else:
            state = (y*self.grid_cols)+ x
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

    def generate_transition_prob(self):
        transition_prob = np.zeros(shape=(self.num_states, self.num_actions, self.num_states))
        next_s = []
        for s in range(self.num_states):
            state = []
            x, y = self.state_to_pos(s)
            for a in range(self.num_actions):
                next = []
                if a == 0: #left
                    next.append(self.pos_to_state(x-1, y))
                    next.append(self.pos_to_state(x-1, y+1))
                    next.append(self.pos_to_state(x-1, y-1))
                elif a == 1: #down
                    next.append(self.pos_to_state(x, y+1))
                    next.append(self.pos_to_state(x+1,y+1))
                    next.append(self.pos_to_state(x-1,y+1))
                elif a == 2: #right
                    next.append(self.pos_to_state(x+1,y))
                    next.append(self.pos_to_state(x+1,y+1))
                    next.append(self.pos_to_state(x+1,y-1))
                elif a == 3: #up
                    next.append(self.pos_to_state(x,y-1))
                    next.append(self.pos_to_state(x+1,y-1))
                    next.append(self.pos_to_state(x-1,y-1))

                next = list(filter(lambda x: x != -1, next))
                if len(next) == 0: next.append(s)
                state.append(next)
        
            next_s.append(state)
                
        for s in range(self.num_states):
            for a in range(self.num_actions):
            
                if len(next_s[s][a]) >1 :
                    str_s = next_s[s][a][0]
                    str_prob = 1
                    prob = (1.0-str_prob)/(len(next_s[s][a])-1)

                    transition_prob[s][a][str_s] = str_prob
                    for next in next_s[s][a][1:]:
                        transition_prob[s][a][next] = prob

                else:
                    transition_prob[s][a][s] = 1.0
                 
        return transition_prob
    
    def visualize_path(self, policy):
        ########
        init_return = self.env.reset()
        state = init_return[0]
        self.env.render()

        count = 0
        while(1):
            old_state = state
            action = policy[state]
            run_return = self.env.step(action)
            state = run_return[0]
            self.env.render()
            count += 1
            time.sleep(0.5)
            if run_return[2] or count>30 or old_state==state:
                break