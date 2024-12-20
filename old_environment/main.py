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

# Check if the environment adheres to the Gym API
check_env(env, warn=True)

# Instantiate the PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=5000000)  # Adjust the timesteps as needed

# Save the trained model
model.save("ppo_skid_steer_model")

