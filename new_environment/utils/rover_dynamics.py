import numpy as np
from scipy.spatial import KDTree
import pandas as pd

def sigmoid(x):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-x))

def bilinear_interpolation(df, x, y):

    if abs(x) > 10 or abs(y) > 10:
        return 0.1, 0.1
    """
    Perform bilinear interpolation to find the z and kappa values at a given (x, y) position.
    """
    df_sorted_x = df.iloc[(df['x'] - x).abs().argsort()]
    x1, x2 = df_sorted_x['x'].unique()[:2]

    df_x1 = df_sorted_x[df_sorted_x['x'] == x1]
    df_x2 = df_sorted_x[df_sorted_x['x'] == x2]

    df_x1_sorted_y = df_x1.iloc[(df_x1['y'] - y).abs().argsort()]
    df_x2_sorted_y = df_x2.iloc[(df_x2['y'] - y).abs().argsort()]

    y1, y2 = df_x1_sorted_y['y'].unique()[:2]

    z11 = df_x1_sorted_y[df_x1_sorted_y['y'] == y1]['z'].values[0]
    z21 = df_x2_sorted_y[df_x2_sorted_y['y'] == y1]['z'].values[0]
    z12 = df_x1_sorted_y[df_x1_sorted_y['y'] == y2]['z'].values[0]
    z22 = df_x2_sorted_y[df_x2_sorted_y['y'] == y2]['z'].values[0]

    kappa11 = df_x1_sorted_y[df_x1_sorted_y['y'] == y1]['kappa'].values[0]
    kappa21 = df_x2_sorted_y[df_x2_sorted_y['y'] == y1]['kappa'].values[0]
    kappa12 = df_x1_sorted_y[df_x1_sorted_y['y'] == y2]['kappa'].values[0]
    kappa22 = df_x2_sorted_y[df_x2_sorted_y['y'] == y2]['kappa'].values[0]

    z_interp = (
        (z11 * (x2 - x) * (y2 - y) +
         z21 * (x - x1) * (y2 - y) +
         z12 * (x2 - x) * (y - y1) +
         z22 * (x - x1) * (y - y1))
        / ((x2 - x1) * (y2 - y1))
    )

    kappa_interp = (
        (kappa11 * (x2 - x) * (y2 - y) +
         kappa21 * (x - x1) * (y2 - y) +
         kappa12 * (x2 - x) * (y - y1) +
         kappa22 * (x - x1) * (y - y1))
        / ((x2 - x1) * (y2 - y1))
    )

    return z_interp, kappa_interp

def skid_steer_dynamics(state, u_lin, u_ang, z, kappa):
    """
    Update rover dynamics based on z and kappa.
    """
    x, y, theta = state
    Radius = 0.09  # Radius of the wheels
    B = 0.25       # Distance between wheels
    dt = 0.1       # Time step

    v = (u_lin * Radius * np.tanh(z) + 3 * sigmoid(kappa))*2
    omega = u_ang * (Radius / B) * np.tanh(z) * sigmoid(kappa) * 3

    theta_new = theta + omega * dt
    x_new = x + v * np.cos(theta_new) * dt
    y_new = y + v * np.sin(theta_new) * dt

    theta_new = np.arctan2(np.sin(theta_new), np.cos(theta_new))
    return [x_new, y_new, theta_new]

def skid_steer_dynamics_with_obstacles(state, u_lin, u_ang, df, obstacle_tree):
    """
    Update rover dynamics while checking for collisions.
    """
    x, y, theta = state
    z, kappa = bilinear_interpolation(df, x, y)
    new_state = skid_steer_dynamics(state, u_lin, u_ang, z, kappa)

    if check_path_collision_with_kdtree(state, new_state, obstacle_tree):
        print("Collision detected! Movement restricted.")
        return state  # Stay in the current state if a collision occurs

    return new_state

def check_collision_with_kdtree(vehicle_state, obstacle_tree, vehicle_length=1.0, vehicle_width=0.5):
    """
    Optimized collision check using KDTree.
    """
    x, y, theta = vehicle_state

    half_length = vehicle_length / 2
    half_width = vehicle_width / 2

    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    corners = np.array([
        [ half_length,  half_width],
        [ half_length, -half_width],
        [-half_length, -half_width],
        [-half_length,  half_width]
    ])

    global_corners = (rotation_matrix @ corners.T).T + np.array([x, y])

    for corner in global_corners:
        neighbors = obstacle_tree.query_ball_point(corner, r=0.1)
        if len(neighbors) > 0:
            return True
    return False

def check_path_collision_with_kdtree(start_state, end_state, obstacle_tree, num_points=10, vehicle_length=1.0, vehicle_width=0.5):
    """
    Optimized path collision check using KDTree.
    """
    x_start, y_start, theta_start = start_state
    x_end, y_end, theta_end = end_state

    x_points = np.linspace(x_start, x_end, num_points)
    y_points = np.linspace(y_start, y_end, num_points)
    theta_points = np.linspace(theta_start, theta_end, num_points)

    for x, y, theta in zip(x_points, y_points, theta_points):
        if check_collision_with_kdtree([x, y, theta], obstacle_tree, vehicle_length, vehicle_width):
            return True
    return False
