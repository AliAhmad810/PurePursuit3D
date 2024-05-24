"""
catmull-rom.py

A Python module for computing Catmull-Rom splines.This module provides functionality 
for generating Catmull-Rom splines from a sequence of control points.

Math retrieved from: https://qroph.github.io/2018/07/30/smooth-paths-using-catmull-rom-splines.html

"""
import numpy as np
import matplotlib.pyplot as plt

__all__ = ["CatmullRom"]

class CatmullRom :
    
    @staticmethod
    def num_segments(points: tuple) -> int:
        """
        Returns the amount of segments included in the point chain. There are 
        n-1 segments for n points, and 2 of them are "ghost segments", so we
        subtract 3.
        :param points: List of control points.
        :return: The number of segments in the spline.
        """
        return len(points) - 3
    
    @staticmethod
    def tj(ti: float, Pi: tuple, Pj: tuple, alpha:float) -> float:
        """
        Returns the value of the next t depending on the distance between current
        p and next p, as well as some alpha to determine knot parameterization.
        :param ti: The current t.
        :param Pi: The current control point.
        :param Pj: The next control point.
        :param alpha: 0.0 for uniform spline, 0.5 for centripetal spline, 1.0
            for chordal spline.
        :return: The value of the next t.
        """
        return ((((Pi[0] - Pj[0]) ** 2) + ((Pi[1] - Pj[1]) ** 2) + ((Pi[2] - Pj[2]) ** 2)) ** 0.5) ** alpha + ti
    
    def spline(
        self,
        P0: tuple, 
        P1: tuple, 
        P2: tuple, 
        P3: tuple, 
        alpha: float, 
        num_points: int
    ) -> list[tuple[float, float, float]]:
        """
        Returns points that make up the spline segment.
        :param P0, P1, P2, P3: The points that define the Catmull-Rom spline.
        :param alpha: 0.0 for uniform spline, 0.5 for centripetal spline, 1.0
            for chordal spline.
        :param num_points: Number of points that should make up the spline.
        :return: List of points that make up the spline.
        """
        t0 : float = 0.0
        t1 : float = self.tj(t0, P0, P1, alpha)
        t2 : float = self.tj(t1, P1, P2, alpha)
        t3 : float = self.tj(t2, P2, P3, alpha)
        t = np.linspace(t1, t2, num_points).reshape(num_points, 1)

        A1 = ((t1 - t) / (t1 - t0)) * np.array(P0) + ((t - t0) / (t1 - t0)) * np.array(P1)
        A2 = ((t2 - t) / (t2 - t1)) * np.array(P1) + ((t - t1) / (t2 - t1)) * np.array(P2)
        A3 = ((t3 - t) / (t3 - t2)) * np.array(P2) + ((t - t2) / (t3 - t2)) * np.array(P3)

        B1 = ((t2 - t) / (t2 - t0)) * A1 + ((t - t0) / (t2 - t0)) * A2
        B2 = ((t3 - t) / (t3 - t1)) * A2 + ((t - t1) / (t3 - t1)) * A3

        C = ((t2 - t) / (t2 - t1)) * B1 + ((t - t1) / (t2 - t1)) * B2

        return C.tolist()

    def spline_chain(self, points: tuple, alpha : float, num_points: int) -> list :
        """
        Calculate Catmull-Rom spline chain for a sequence of control points.
        :param points: Control points.
        :param alpha: Alpha parameter for knot spacing.
        :param num_points: The number of points to include in each.
        :return: The chain of all points that make up the full spline.
        """
        point_quadruples = [
            points[i:i + 4] for i in range(self.num_segments(points))
        ]
        splines = [
            self.spline(*p_list, alpha, num_points) for p_list in point_quadruples
        ]
        return [pt for spline in splines for pt in spline]

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