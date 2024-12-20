from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from utils.gym_env import SkidSteerEnv
import pandas as pd
from scipy.spatial import KDTree

env_df = pd.read_csv('utils/environment.csv')
obstacle_positions = env_df[env_df['is_obstacle']][['x', 'y']].values
obstacle_tree = KDTree(obstacle_positions)

# Create the environment
env = SkidSteerEnv(df=env_df, obstacle_tree=obstacle_tree, goal=(2.5, 10))

model = PPO.load("ppo_skid_steer_model")

trajectory = []  # To store the trajectory of the agent

obs, _ = env.reset()
done = False
truncated = False

while not done and not truncated:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    #print(env.state)
    # Append current position to the trajectory
    trajectory.append(env.state[:2])  # Assuming state = [x, y, theta]

#print(trajectory)
# Convert trajectory to a NumPy array for plotting
import numpy as np
trajectory = np.array(trajectory)
# save trajectory to a CSV file
pd.DataFrame(trajectory, columns=['x', 'y']).to_csv('output_data/trajectory.csv', index=False)