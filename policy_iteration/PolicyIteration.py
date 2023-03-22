import numpy as np
import matplotlib.pyplot as plt
import time

class PolicyIteration:
    def __init__(self, World, gamma, init_policy=None):
        self.world = World
        self.num_states = World.num_states
        self.num_actions = World.num_actions
        self.reward_function = World.reward_map
        self.transition_model = World.transition_prob
        self.gamma = gamma
        self.values = np.zeros(self.num_states)

        #for collecting history
        self.evaluation_iteration = []
        self.history_policy = []
        self.history_values = []
        self.history_values.append([0 for i in range(self.num_states)])

        if init_policy is None:
            self.policy = np.random.choice(self.world.action_space, size=self.num_states)
            policy = [a for a in self.policy]
            self.history_policy.append(policy)
        else:
            self.policy = init_policy

    def policy_evaluation(self):
        delta = 0
        for s in range(self.num_states):
            x, y = self.world.state_to_pos(s)
            temp = self.values[s]
            a = self.policy[s]
            p = self.world.transition_prob[s][a]
            self.values[s] = self.reward_function[y][x] + self.gamma * np.sum(p * self.values)
            delta = max(delta, abs(temp - self.values[s]))
        return delta
    
    def policy_evaluation_run(self, threshold=0.0005):
        converged = False
        count_evalation = 0
        while(not converged):
            delta = self.policy_evaluation()
            count_evalation += 1
            if (delta < threshold):
                converged = True
        return count_evalation
        # self.evaluation_iteration.append(count_evalation)
    
    def argmax(self, s):
        action_value = np.zeros(self.num_actions)
        for a in range(self.num_actions):
            p = self.world.transition_prob[s][a]
            action_value[a] = np.sum(p * self.values)
        return np.argmax(action_value)
       
    def policy_improvement(self):
        stable = True
        for s in range(self.num_states):
            old_a = self.policy[s]
            self.policy[s] = self.argmax(s)
            a = self.policy[s]
            stable = stable and (old_a == a)
        return stable

    def policy_iteration_solve(self):
        policy_stable = False
        
        while(not policy_stable):
            evaluation_count = self.policy_evaluation_run()
            policy_stable = self.policy_improvement()

            self.evaluation_iteration.append(evaluation_count)
            policy = [a for a in self.policy]
            values = [v for v in self.values]

            #collecting policy and values of each iteration
            self.history_policy.append(policy)
            self.history_values.append(values)
            
    def result_visualize(self):
        # fig, ax = plt.subplots()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        policy_table = []
        colors_table = []
        
        for r in range(self.world.grid_rows):
            row = []
            color_cell = []
            for c in range(self.world.grid_cols):
                s = self.world.pos_to_state(c, r)

                action = self.world.action_dict[self.policy[s]]
                # text = "{} \n {:.1f} [S{}]".format(action, self.values[s], s)
                text = "{} \n S{}".format(action, s)
                row.append(text)
                
                color = self.world.env_color[self.world.map[r][c]]
                color_cell.append(color)
            policy_table.append(row)
            colors_table.append(color_cell)

        ax1.set_axis_off()
        ax1.title.set_text('Policy')
        ax1.set_box_aspect(True)
        myTable = ax1.table(cellText=policy_table, cellColours=colors_table, cellLoc='center', rowLoc='center', loc='center')
        myTable.auto_set_font_size(True)
        myTable.scale(1, 1)

        cellDict = myTable.get_celld()
        for cell in cellDict:
            cellDict[cell].set_height(0.1)

            # ax2.set_axis_off()
        ax2.title.set_text('Policy Evaluation')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Sweeps')
        ax2.set_box_aspect(True)

        ax2.set_xticks(range(len(self.evaluation_iteration)), range(len(self.evaluation_iteration)))
        ax2.plot(range(len(self.evaluation_iteration)), self.evaluation_iteration, 'ro-')

        plt.show()

    def state_value_visualize(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        ax1.set_axis_off()
        ax1.title.set_text('State Value')
        ax1.set_box_aspect(True)

        ax2.set_axis_off()
        ax2.title.set_text('Policy')
        ax2.set_box_aspect(True)

        for i in range(len(self.history_policy)):
            policy_table = []
            values = np.zeros((self.world.grid_rows, self.world.grid_cols))

            for r in range(self.world.grid_rows):
                row = []
                for c in range(self.world.grid_cols):
                    s = self.world.pos_to_state(c, r)

                    action = self.world.action_dict[self.history_policy[i][s]]
                    # text = "{} \n {:.3f}".format(action, self.history_values[i][s])
                    row.append(action)

                    values[r][c] = self.history_values[i][s]

                policy_table.append(row)

            abs_min = abs(values.min())
            norm = plt.Normalize(np.log(2), np.log(values.max()+abs_min+20))
            colors_table = plt.cm.hot(norm(np.log(values+abs_min+20)))

            ValueTable = ax1.table(cellText=np.around(values, decimals=3), 
                                cellColours=colors_table, 
                                cellLoc='center', rowLoc='center', loc='center')
            ValueTable.auto_set_font_size(False)
            ValueTable.set_fontsize(6)
            ValueTable.scale(1, 1)

            PolicyTable = ax2.table(cellText=policy_table, 
                                cellColours=colors_table, 
                                cellLoc='center', rowLoc='center', loc='center')
            PolicyTable.auto_set_font_size(False)
            PolicyTable.set_fontsize(8)
            PolicyTable.scale(1, 1)

            cellDict1 = ValueTable.get_celld()
            cellDict2 = PolicyTable.get_celld()
            for cell in cellDict1:
                cellDict1[cell].set_height(0.1)
                cellDict2[cell].set_height(0.1)

            plt.pause(1)
        
        plt.show()