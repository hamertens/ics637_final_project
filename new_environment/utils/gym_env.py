import gymnasium as gym
from gymnasium import spaces
import numpy as np
from utils.rover_dynamics import skid_steer_dynamics_with_obstacles, bilinear_interpolation, sigmoid


def linear_interpolation(actual_value, min_val, max_val):
    """
    Linearly interpolates the actual value based on a specified range.

    Parameters:
        actual_value (float): The value to interpolate.
        min_val (float): The minimum value, mapped to 0.
        max_val (float): The maximum value, mapped to 1.

    Returns:
        float: The interpolated value between 0 and 1.

    Raises:
        ValueError: If min_val is equal to max_val (division by zero).
    """
    if min_val == max_val:
        raise ValueError("Minimum and maximum values cannot be the same.")

    # Perform linear interpolation
    interpolated_value = (actual_value - min_val) / (max_val - min_val)
    return interpolated_value

class SkidSteerEnv(gym.Env):
    def __init__(self, df, obstacle_tree, goal):
        """
        Initialize the environment.
        
        Parameters:
        - df: DataFrame containing the environment grid (for z and kappa interpolation).
        - obstacle_tree: KDTree for obstacle collision checking.
        - goal: Tuple (x, y) representing the goal position.
        """
        super(SkidSteerEnv, self).__init__()
        
        # Environment parameters
        self.df = df
        self.obstacle_tree = obstacle_tree
        self.goal = goal

        self.max_steps = 80
        self.y_reward_scale = 2
        self.speed_reward_scale = 3
        
        
        # State: [x, y, theta]
        self.state = None
        
        # Action space: [u_lin, u_ang]
        self.action_space = spaces.Box(
            low=np.array([-50.0, -25.0]),  # Minimum values for u_lin and u_ang
            high=np.array([50.0, 25.0]),   # Maximum values for u_lin and u_ang
            dtype=np.float32
        )
        
        # Observation space: [x, y, theta]
        self.observation_space = spaces.Box(
            low=np.array([-20, -20, -np.pi]),  # Min values for x, y, theta
            high=np.array([20, 20, np.pi]),    # Max values for x, y, theta
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """
        Reset the environment to the initial state.
        
        Parameters:
        - seed: Random seed for reproducibility.
        - options: Optional dictionary for additional reset options.
        
        Returns:
        - observation: The initial observation of the environment.
        - info: A dictionary with additional information (empty for now).
        """
        # Set the seed for reproducibility
        super().reset(seed=seed)
        self.np_random, seed = gym.utils.seeding.np_random(seed)

        # Reset the environment state
        self.state = [5.0, -10.0, np.pi/2]  # Example starting state
        self.steps_taken = 1  # Reset step counter

        # Return initial observation and info
        return self._get_observation(), {}

    def step(self, action):
        """
        Take a step in the environment.

        Parameters:
        - action: [u_lin, u_ang] (linear and angular velocities)

        Returns:
        - obs: Observation after taking the action
        - reward: Reward for this step
        - terminated: Whether the episode is terminated (goal reached, etc.)
        - truncated: Whether the episode is truncated (time limit exceeded)
        - info: Additional info (empty for now)
        """
        # Extract linear and angular velocity from the action
        self.u_lin, self.u_ang = action  # Store the action as environment attributes
        
        # Update state using the provided dynamics with obstacle handling
        new_state = skid_steer_dynamics_with_obstacles(self.state, self.u_lin, self.u_ang, self.df, self.obstacle_tree)
        self.state = new_state

        # Calculate reward
        reward = self._calculate_reward()

        # Check termination conditions
        terminated = bool(self._is_done())

        # Check truncation condition (e.g., time limit exceeded)
        truncated = self.steps_taken >= self.max_steps  # Set self.max_steps in __init__()

        # Increment step counter
        self.steps_taken += 1

        # Return the required values
        return self._get_observation(), reward, terminated, truncated, {}

    def _get_observation(self):
        """
        Return the current state as an observation.
        """
        # Observation now only includes x, y, and theta
        observation = np.array(self.state, dtype=np.float32)
        return observation

    def _calculate_reward(self):
        """
        Calculate the reward for the current step.
        """
        x, y, _ = self.state
        distance_to_goal = np.linalg.norm([self.goal[0] - x, self.goal[1] - y])
        
        # Base reward: Negative distance to the goal
        reward = -distance_to_goal

        reward += self.y_reward_scale*y

        # Cumulative time penalty
        #reward -= 0.1 * self.steps_taken  # Penalty increases with each time step

        z_min = self.df['z'].min()
        z_max = self.df['z'].max()
        kappa_min = self.df['kappa'].min()
        kappa_max = self.df['kappa'].max()
        z, kappa = bilinear_interpolation(self.df, x, y)
        scaled_kappa = linear_interpolation(kappa, kappa_min, kappa_max)
        scaled_z = linear_interpolation(z, z_min, z_max)
        
        reward += self.speed_reward_scale * (scaled_kappa + scaled_z)  # Reward for moving faster

        # Collision penalty
        if np.allclose(self.state, skid_steer_dynamics_with_obstacles(self.state, 0, 0, self.df, self.obstacle_tree)):
            reward -= 100  # Large penalty for collisions (staying in the same state)
        
        if abs(x) > 10 or abs(y) > 10:
            reward -= 300  # Large penalty for going out of bounds

        if distance_to_goal < 0.5:
            reward += 2000 / self.steps_taken

        if distance_to_goal < 1.1:  # Threshold for getting close to the goal
            reward += 2000 / self.steps_taken

        if distance_to_goal < 2.1:  # Threshold for getting close to the goal
            reward += 1000 / self.steps_taken

        if distance_to_goal < 3.1:  # Threshold for getting close to the goal
            reward += 500 / self.steps_taken

        if distance_to_goal < 4.1:  # Threshold for getting close to the goal
            reward += 200 / self.steps_taken

        if distance_to_goal < 5.1:  # Threshold for getting close to the goal
            reward += 100 / self.steps_taken      

        # Goal bonus scaled by time
        if distance_to_goal < 0.1:  # Threshold for reaching the goal
            reward += 10000000 / self.steps_taken  # Larger bonus for quicker goal achievement

        # Increment steps taken for scaling the time penalty and goal bonus
        #self.steps_taken += 1

        return reward

    def _is_done(self):
        """Check if the episode is done."""
        x, y, _ = self.state
        distance_to_goal = np.linalg.norm([self.goal[0] - x, self.goal[1] - y])
        return distance_to_goal < 0.1  # Episode ends if the goal is reached

    def render(self, mode="human"):
        """Optional: Render the environment."""
        print(f"State: {self.state}")
