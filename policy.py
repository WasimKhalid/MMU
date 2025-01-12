import numpy as np
import random

# Parameters for Q-learning
ALPHA = 0.1  # 
GAMMA = 0.9  
PSILON = 0.7 

class DisasterEnvironment:
    def __init__(self, grid_size, num_survivors, num_hazards):
        self.grid_size = grid_size
        self.num_survivors = num_survivors
        self.num_hazards = num_hazards
        self.reset()

    def reset(self):
        #Reset the environment to its initial state
        self.grid = np.zeros((self.grid_size, self.grid_size))
        self.robot_pos = [0, 0]  
        self.path = [tuple(self.robot_pos)]
        self.survivors = set()
        self.hazards = set()
        self.place_items()  
        return self.robot_pos

    def place_items(self):
        """Randomly place survivors and hazards in the environment."""
        while len(self.survivors) < self.num_survivors:
            pos = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            if pos != (0, 0):  
                self.survivors.add(pos)
        while len(self.hazards) < self.num_hazards:
            pos = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            if pos != (0, 0) and pos not in self.survivors:
                self.hazards.add(pos)

    def step(self, action):
        """Perform an action and return the new state, reward, and whether the episode is done."""
        moves = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1)    # right
        }
        move = moves[action]
        new_pos = [self.robot_pos[0] + move[0], self.robot_pos[1] + move[1]]

        
        if 0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size:
            self.robot_pos = new_pos
            self.path.append(tuple(self.robot_pos))

        # Reward and penalties
        reward = -1 
        if tuple(self.robot_pos) in self.survivors:
            reward = 10  # Reward for rescuing a survivor
            self.survivors.remove(tuple(self.robot_pos))  
        elif tuple(self.robot_pos) in self.hazards:
            reward = -5  

        done = len(self.survivors) == 0 
        return self.robot_pos, reward, done


class QLearningAgent:
    def __init__(self, env):
        self.q_table = np.zeros((env.grid_size, env.grid_size, 4))  
        self.env = env

    def choose_action(self, state, epsilon):
        """Choose an action based on epsilon-greedy policy."""
        if random.uniform(0, 1) < epsilon:  
            return random.choice(range(4))  
        else:  # Exploitation
            x, y = state
            return np.argmax(self.q_table[x, y])  

    def update_q_value(self, state, action, reward, next_state):
        
        x, y = state
        next_x, next_y = next_state
        old_value = self.q_table[x, y, action]
        next_max = np.max(self.q_table[next_x, next_y])  
        new_value = old_value + ALPHA * (reward + GAMMA * next_max - old_value)  # Update Q-value
        self.q_table[x, y, action] = new_value

    def decay_epsilon(self, epsilon, decay_rate=0.99):
      
        return epsilon * decay_rate
