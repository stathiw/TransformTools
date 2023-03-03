#  Set up a gui class which creates a 3d plot using plotly

import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# import classes from geometry.py
from geometry import *


class Gui:
    def __init__(self):
        self.fig = px.scatter_3d(x=[0], y=[0], z=[0], color_discrete_sequence=['red'])
        self.fig.update_traces(marker=dict(size=1, line=dict(width=0, color='DarkSlateGrey')))

    def draw_line(self, x1, y1, z1, x2, y2, z2, color, width):
        """ Draw a line between two points """
        self.fig.add_trace(
            go.Scatter3d(
                x=[x1, x2],
                y=[y1, y2],
                z=[z1, z2],
                mode='lines',
                line=dict(color=color, width=width)
            )
        )

    def draw_line_between_points(self, p1, p2, color, width):
        """ Draw a line between two points """
        self.draw_line(p1.x, p1.y, p1.z, p2.x, p2.y, p2.z, color, width)

    def draw_axis(self, pose, length, width):
        """ Draw the axis of a pose """
        origin = pose.t
        # transform the origin to the pose frame
        p1 = pose.transform_from(Point3(0, 0, 0))
        p2_x = pose.transform_from(Point3(length, 0, 0))
        p2_y = pose.transform_from(Point3(0, length, 0))
        p2_z = pose.transform_from(Point3(0, 0, length))

        self.draw_line_between_points(p1, p2_x, 'red', width)
        self.draw_line_between_points(p1, p2_y, 'green', width)
        self.draw_line_between_points(p1, p2_z, 'blue', width)

    def plot(self, x, y, z, color):
        """ Plot a point """
        self.fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='markers',
                marker=dict(
                    size=2,
                    color=color,
                    opacity=0.8
                )
            )
        )

    def show(self):
        """ Show the plot """
        self.fig.show()

# Path: main.py
#  Set up a main class which creates a 3d plot using plotly

if __name__ == '__main__':
    gui = Gui()

    # Draw axis at origin
    gui.draw_axis(Pose3(), 1, 5)

    # Draw another axis at 5, 5, 5
    R = Rot3.RPY(0, 0, np.pi / 4)
    t = Point3(5, 5, 5)
    gui.draw_axis(Pose3(R, t), 1, 5)

    gui.show()
