import numpy as np
from einops import einsum, rearrange
from physics.rotation import rotmat

import jax
from jax import numpy as jnp


class Positions:
    def __init__(
        self,
        body_polygon_x,
        body_polygon_y,
        hip_pos,
        lknee_pos,
        lfoot_pos,
        rknee_pos,
        rfoot_pos,
    ):
        self.body_polygon_x = body_polygon_x
        self.body_polygon_y = body_polygon_y
        self.hip_pos = hip_pos
        self.lknee_pos = lknee_pos
        self.lfoot_pos = lfoot_pos
        self.rknee_pos = rknee_pos
        self.rfoot_pos = rfoot_pos


def flatten_positions(positions):
    return [
        positions.body_polygon_x,
        positions.body_polygon_y,
        positions.hip_pos,
        positions.lknee_pos,
        positions.lfoot_pos,
        positions.rknee_pos,
        positions.rfoot_pos,
    ], None


def unflatten_positions(aux_data, flat_positions):
    return Positions(
        body_polygon_x=flat_positions[0],
        body_polygon_y=flat_positions[1],
        hip_pos=flat_positions[2],
        lknee_pos=flat_positions[3],
        lfoot_pos=flat_positions[4],
        rknee_pos=flat_positions[5],
        rfoot_pos=flat_positions[6],
    )


jax.tree_util.register_pytree_node(
    Positions,
    flatten_positions,
    unflatten_positions,
)


def make_positions(q, shape_config):
    body_theta = q[0]
    body_pos = q[1:3]
    body_len = shape_config[0]
    thigh_len = shape_config[1]
    shank_len = shape_config[2]

    body_polygon_base = jnp.array(
        [
            [-body_len / 2, -body_len / 2],
            [body_len / 2, -body_len / 2],
            [body_len / 2, body_len / 2],
            [-body_len / 2, body_len / 2],
            [-body_len / 2, -body_len / 2],
        ],
        dtype=np.float32,
    )

    body_polygon_now = (
        einsum(rotmat(q[0]), body_polygon_base, "i j, V j -> V i") + body_pos
    )
    (body_polygon_x, body_polygon_y) = rearrange(body_polygon_now, "v a -> a v")

    hip_pos = body_pos + rotmat(body_theta) @ jnp.array(
        [0.0, -body_len / 2.0], dtype=np.float32
    )
    lknee_pos = hip_pos + rotmat(body_theta + q[3]) @ jnp.array(
        [0.0, -thigh_len], dtype=np.float32
    )
    lfoot_pos = lknee_pos + rotmat(body_theta + q[3] + q[4]) @ jnp.array(
        [0.0, -shank_len], dtype=np.float32
    )

    rknee_pos = hip_pos + rotmat(body_theta + q[5]) @ jnp.array(
        [0.0, -thigh_len], dtype=np.float32
    )
    rfoot_pos = rknee_pos + rotmat(body_theta + q[5] + q[6]) @ jnp.array(
        [0.0, -shank_len], dtype=np.float32
    )

    return Positions(
        body_polygon_x=body_polygon_x,
        body_polygon_y=body_polygon_y,
        hip_pos=hip_pos,
        lknee_pos=lknee_pos,
        lfoot_pos=lfoot_pos,
        rknee_pos=rknee_pos,
        rfoot_pos=rfoot_pos,
    )
