"""
Plot.py

A Python module for plotting the graphs and animations for path display
"""

import numpy as np
import matplotlib.pyplot as plt
from PathGen import CatmullRom

def generate_random_pts():
    """
    Returns a list of a random amount of random points.
    :return: Points.
    """

    points = []
    length = np.random.randint(5, 7)
    for _ in range(length):
        point = [np.random.randint(0, 12), np.random.randint(0, 12), np.random.randint(0, 12)]
        points.append(point)
    
    # Convert the list of points to a NumPy array
    points_array = np.array(points)
    return points_array


if __name__ == "__main__" :
    control_points = generate_random_pts()
    catmull_rom = CatmullRom()
    curve = catmull_rom.spline_chain(control_points, alpha=0.5, num_points=100)

    # Plot the result
    curve_np = np.array(curve)
    control_points_np = np.array(control_points)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(curve_np[:, 0], curve_np[:, 1], curve_np[:, 2], label="Catmull-Rom Spline")
    ax.plot(control_points_np[:, 0], control_points_np[:, 1], control_points_np[:, 2], 'ro', label="Control Points")
    plt.legend()
    plt.show()