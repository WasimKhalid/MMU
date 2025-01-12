import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation
from tkinter import PhotoImage

# Function to start the simulation based on user input
def start_simulation():
    try:
        # Get user inputs
        num_survivors = int(survivor_entry.get())
        num_hazards = int(hazard_entry.get())
        num_episodes = int(episode_entry.get())
        
        # Validate inputs
        if num_survivors <= 0 or num_hazards <= 0 or num_episodes <= 0:
            raise ValueError("All inputs must be positive integers.")

        # Set grid size (fixed)
        GRID_SIZE = 10  # Grid size (10x10)

        # Initialize environment and agent
        env = DisasterEnvironment(GRID_SIZE, num_survivors, num_hazards)
        agent = QLearningAgent(env)

        # Start the simulation by passing the environment, agent, and number of episodes
        animate_robot(env, agent, num_episodes)  # Run for the user-specified number of episodes

        # Close the window after simulation starts
        root.destroy()
    
    except ValueError as e:
        # Show an error message if input is invalid
        messagebox.showerror("Invalid input", str(e))

# Function to animate the robot and simulate the environment
def animate_robot(env, agent, episodes=1):  # Accept the episodes argument
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(-0.5, env.grid_size - 0.5)
    ax.set_ylim(-0.5, env.grid_size - 0.5)
    plt.title("Robot Assisted Disaster Response Simulation")

    # Load icons here inside the animate_robot function
    robot_icon_path = 'robot_icon.png'
    survivor_icon_path = 'survivor_icon.png'
    hazard_icon_path = 'hazard_icon.png'

    # Load the icons using PhotoImage for proper display in Tkinter and figimage
    robot_icon = PhotoImage(file=robot_icon_path)
    survivor_icon = PhotoImage(file=survivor_icon_path)
    hazard_icon = PhotoImage(file=hazard_icon_path)

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

    # Performance metrics
    total_rewards = 0  # Track total rewards
    total_cost = 0  # Track total cost
    total_steps = 0  # Track the number of steps

    def update(frame):
        nonlocal total_rewards, total_cost, total_steps
        state = env.robot_pos
        action = agent.choose_action(state, 0.1)  # Use epsilon value here (0.1)
        next_state, reward, done = env.step(action)

        # Update Q-value based on action taken
        agent.update_q_value(state, action, reward, next_state)

        # Update metrics
        total_rewards += reward  # Add reward
        if reward == -5:  # If penalty (cost)
            total_cost += abs(reward)  # Track cost as positive value
        total_steps += 1  # Increase step count

        # Clear and redraw the plot at each frame for reliable update of elements
        ax.clear()
        ax.set_xlim(-0.5, env.grid_size - 0.5)
        ax.set_ylim(-0.5, env.grid_size - 0.5)
        plt.title("Robot Assisted Disaster Response Simulation")

        # Add gridlines for 10x10 grid
        ax.set_xticks(range(env.grid_size))
        ax.set_yticks(range(env.grid_size))
        ax.grid(True, which='both', color='black', linestyle='-', linewidth=1.5)

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
            env.reset()  # Reset the environment after all survivors are rescued

        # If done (all survivors rescued), display performance metrics and stop the animation
        if done:
            print("\nSimulation Finished!")
            print(f"Total Steps Taken: {total_steps}")
            print(f"Total Rewards: {total_rewards}")
            print(f"Total Cost: {total_cost}")
            print(f"Average Reward per Step: {total_rewards / total_steps:.2f}")
            print(f"Average Cost per Step: {total_cost / total_steps:.2f}")
            
            plt.close(fig)  # Close the plot window immediately after one episode
            return None  # Stop the animation loop

        return ax

    # Increase speed by reducing the interval time (milliseconds between frames)
    ani = animation.FuncAnimation(fig, update, frames=episodes * env.grid_size**2, interval=10, blit=False)  # interval=10ms for super-fast simulation
    plt.show()

    # Add simple labels below the grid (using `fig.text` to place text below the plot)
    fig.text(0.5, -0.05, 'Robot Icon', ha='center', va='center', fontsize=12)
    fig.text(0.5, -0.1, 'Hazard Icon', ha='center', va='center', fontsize=12)
    fig.text(0.5, -0.15, 'Path Taken', ha='center', va='center', fontsize=12)

    # Add robot icon image below the grid
    fig.figimage(robot_icon, 0.5, -0.2, zorder=1)
    # Add hazard icon image below the grid
    fig.figimage(hazard_icon, 0.5, -0.3, zorder=1)
    # Optionally add the path image
    fig.figimage(survivor_icon, 0.5, -0.4, zorder=1)

    # Adjust spacing between plot and labels
    plt.subplots_adjust(bottom=0.3)
