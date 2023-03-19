s):
            for c in range(self.grid_cols):
                reward = self.reward[self.map[r, c]]
                reward_map[r][c] = reward