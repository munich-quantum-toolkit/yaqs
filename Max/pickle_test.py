with open('trajectories_timesteps_10.pkl', 'rb') as f:
    loaded_data = pickle.load(f)

# Access the saved components
all_trajectories = loaded_data['all_trajectories']
result_lindblad_interp = loaded_data['result_lindblad_interp']
time_points = loaded_data['time_points']

print("Trajectories:", all_trajectories)
print("Lindblad Interpolation:", result_lindblad_interp)
print("Time Points:", time_points)
