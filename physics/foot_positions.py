import jax
import jax.numpy as jnp


def rotmat(theta):
    return jnp.array(
        [
            [jnp.cos(theta), -jnp.sin(theta)],
            [jnp.sin(theta), jnp.cos(theta)],
        ],
        dtype=jnp.float32,
    )


def foot_pos(body_q, foot_q, shape_config):
    body_theta = body_q[0]
    body_pos = body_q[1:2]

    body_len = shape_config[0]
    thigh_len = shape_config[1]
    shank_len = shape_config[2]

    hip_pos = body_pos + rotmat(body_theta) @ jnp.array(
        [0.0, -body_len / 2.0], dtype=jnp.float32
    )

    knee_pos = hip_pos + rotmat(body_theta + foot_q[0]) @ jnp.array(
        [0.0, -thigh_len], dtype=jnp.float32
    )

    result = knee_pos + rotmat(body_theta + foot_q[0] + foot_q[1]) @ jnp.array(
        [0.0, -shank_len], dtype=jnp.float32
    )

    return result


def lfoot_pos(q, shape_config):
    return foot_pos(q[0:3], q[3:5], shape_config)


def rfoot_pos(q, shape_config):
    return foot_pos(q[0:3], q[5:7], shape_config)
