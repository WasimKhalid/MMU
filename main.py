import tkinter as tk
from tkinter import messagebox
from policy import DisasterEnvironment, QLearningAgent
from simulation import animate_robot  

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

        # Set grid size
        GRID_SIZE = 10  

        # Initialize environment and agent
        env = DisasterEnvironment(GRID_SIZE, num_survivors, num_hazards)
        agent = QLearningAgent(env)

        
        animate_robot(env, agent, num_episodes)  

       
        root.destroy()
    
    except ValueError as e:
        # Show an error message if input is invalid
        messagebox.showerror("Invalid input", str(e))


root = tk.Tk()
root.title("Robotic Disaster Management Simulation")

# Set window size
root.geometry("400x300")


tk.Label(root, text="Please enter Number of Survivors:").pack(pady=10)
survivor_entry = tk.Entry(root)
survivor_entry.pack()

tk.Label(root, text="Please enter Number of Hazards:").pack(pady=10)
hazard_entry = tk.Entry(root)
hazard_entry.pack()

tk.Label(root, text="Number of Episodes:").pack(pady=10)
episode_entry = tk.Entry(root)
episode_entry.pack()

# Add a button to start the simulation
start_button = tk.Button(root, text="Start Simulation", command=start_simulation)
start_button.pack(pady=20)


root.mainloop()
