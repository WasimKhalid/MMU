import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation
import random

# Parameters
GRID_SIZE = 10
NUM_SURVIVORS = random.randint(5,6)
NUM_HAZARDS = random.randint(5,9)
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1
EPISODES = 10

# Icons directory:
robot_icon_path = r'E:\robot_icon.png'
survivor_icon_path = r'E:\survivor_icon.png'
hazard_icon_path = r'E:\hazard_icon.png'
#background_image_path = r"E:/background.png"  # Your grid background image


# images loading 
robot_icon = mpimg.imread(robot_icon_path)
survivor_icon = mpimg.imread(survivor_icon_path)
hazard_icon = mpimg.imread(hazard_icon_path)
#background_icon=mpimg(background_image_path)

# Disaster Environment Setting
class DisasterEnvironment:
    def __init__(self, grid_size, num_survivors, num_hazards):
        self.grid_size = grid_size
        self.num_survivors = num_survivors
        self.num_hazards = num_hazards
        self.reset()

    def reset(self):
        self.grid = np.zeros((self.grid_size, self.grid_size))
        self.robot_pos = [0, 0]
        self.path = [tuple(self.robot_pos)]  # Track robot's path
        self.survivors = set()
        self.hazards = set()
        self.place_items()
        return self.robot_pos

    def place_items(self):
        while len(self.survivors) < self.num_survivors:
            pos = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            if pos != (0, 0):
                self.survivors.add(pos)
        while len(self.hazards) < self.num_hazards:
            pos = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            if pos != (0, 0) and pos not in self.survivors:
                self.hazards.add(pos)

    def step(self, action):
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
            self.path.append(tuple(self.robot_pos))  # path update
        reward = -1
        if tuple(self.robot_pos) in self.survivors:
            reward = 10
            self.survivors.remove(tuple(self.robot_pos))  # Removal of survivor when reached on the spot
        elif tuple(self.robot_pos) in self.hazards:
            reward = -10
        done = len(self.survivors) == 0
        return self.robot_pos, reward, done

# Q-Learning Agent implementation
class QLearningAgent:
    def __init__(self, env):
        self.q_table = np.zeros((env.grid_size, env.grid_size, 4))
        self.env = env

    def choose_action(self, state, epsilon):
        if random.uniform(0, 1) < epsilon:
            return random.choice(range(4))
        else:
            x, y = state
            return np.argmax(self.q_table[x, y])

    def update_q_value(self, state, action, reward, next_state):
        x, y = state
        next_x, next_y = next_state
        old_value = self.q_table[x, y, action]
        next_max = np.max(self.q_table[next_x, next_y])
        new_value = old_value + ALPHA * (reward + GAMMA * next_max - old_value)
        self.q_table[x, y, action] = new_value

# Animated Visualization
def animate_robot(env, agent, episodes=10):
    fig, ax = plt.subplots()
    ax.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax.set_ylim(-0.5, GRID_SIZE - 0.5)
    plt.title("Robot Assisted Disaster Response Simulation")

    # Initialize robot's icon at the starting position
    robot_img = ax.imshow(robot_icon, extent=(env.robot_pos[1]-0.5, env.robot_pos[1]+0.5, env.robot_pos[0]-0.5, env.robot_pos[0]+0.5))
    path_lines, = ax.plot([], [], 'red', linestyle='-', linewidth=1.5, label="Path")

    # Create dictionaries for icons without labels
    survivor_images = {}
    hazard_images = {}
    for pos in env.hazards:
        hazard_images[pos] = ax.imshow(hazard_icon, extent=(pos[1]-0.5, pos[1]+0.5, pos[0]-0.5, pos[0]+0.5))
    for pos in env.survivors:
        survivor_images[pos] = ax.imshow(survivor_icon, extent=(pos[1]-0.5, pos[1]+0.5, pos[0]-0.5, pos[0]+0.5))

    def update(frame):
        state = env.robot_pos
        action = agent.choose_action(state, EPSILON)
        next_state, reward, done = env.step(action)
        agent.update_q_value(state, action, reward, next_state)

        # Clear and redraw the plot at each frame for reliable update of elements
        ax.clear()
        ax.set_xlim(-0.5, GRID_SIZE - 0.5)
        ax.set_ylim(-0.5, GRID_SIZE - 0.5)
        plt.title("Robot Assisted Disaster Response Simulation")

        # Update path trace
        path_x, path_y = zip(*env.path)
        ax.plot(path_y, path_x, 'red', linestyle='-', linewidth=1.5)

        # Update robot's position on the grid
        ax.imshow(robot_icon, extent=(env.robot_pos[1]-0.5, env.robot_pos[1]+0.5, env.robot_pos[0]-0.5, env.robot_pos[0]+0.5))

        # Plot hazards and survivors
        for pos, icon in hazard_images.items():
            ax.imshow(hazard_icon, extent=(pos[1]-0.5, pos[1]+0.5, pos[0]-0.5, pos[0]+0.5))
    
        for pos in env.survivors:
            if pos in survivor_images:
                ax.imshow(survivor_icon, extent=(pos[1]-0.5, pos[1]+0.5, pos[0]-0.5, pos[0]+0.5))

        # Remove survivor icon if robot reaches it
        if tuple(env.robot_pos) in survivor_images:
            del survivor_images[tuple(env.robot_pos)]  # Remove from dictionary
            env.survivors.discard(tuple(env.robot_pos))  # Remove survivor from environment data

        # Reset environment if all survivors are saved
        if done:
            env.reset()
            env.path = [tuple(env.robot_pos)]

        return ax

    ani = animation.FuncAnimation(fig, update, frames=episodes * env.grid_size**2, interval=300)
    plt.show()

# Initialize environment and agent, then run the visualization
env = DisasterEnvironment(GRID_SIZE, NUM_SURVIVORS, NUM_HAZARDS)
agent = QLearningAgent(env)
animate_robot(env, agent, EPISODES)
