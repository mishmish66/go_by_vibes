import matplotlib
matplotlib.use("TKAgg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np

from physics.rotation import rotmat

from einops import einsum, rearrange

import threading

from physics.positions import make_positions


def animate(qs, shape_config, dt=0.01, dpi=100):
    fig = plt.figure(figsize=(1440 / dpi, 1440 / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    (floor_line,) = ax.plot([-100, 100], [-2, -2], lw=3)
    (body_line,) = ax.plot([], [], lw=3)
    (lleg_line,) = ax.plot([], [], lw=3)
    (rleg_line,) = ax.plot([], [], lw=3)

    def update_viz(q):
        positions = make_positions(q, shape_config)

        body_line.set_data(positions.body_polygon_x, positions.body_polygon_y)

        (lleg_x, lleg_y) = rearrange(
            [positions.hip_pos, positions.lknee_pos, positions.lfoot_pos], "v d -> d v"
        )
        lleg_line.set_data(lleg_x, lleg_y)

        (rleg_x, rleg_y) = rearrange(
            [positions.hip_pos, positions.rknee_pos, positions.rfoot_pos], "v d -> d v"
        )
        rleg_line.set_data(rleg_x, rleg_y)

        fig.canvas.draw()
        return body_line, lleg_line, rleg_line

    ani = FuncAnimation(
        fig,
        update_viz,
        interval=dt,
        blit=True,
        frames=qs,
        repeat=True,
    )

    return ani
