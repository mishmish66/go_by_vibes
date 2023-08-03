import jax.numpy as jnp
from jax.lax import cond
from jax import grad, jit, vmap

from typing import Callable


def rot(theta):
    return jnp.array(
        [
            [jnp.cos(theta), -jnp.sin(theta)],
            [jnp.sin(theta), jnp.cos(theta)],
        ]
    )


class State:
    q: jnp.ndarray
    v: jnp.ndarray

    class Config:
        g: float = 9.81
        
        m_body: float = 1.0
        m_thigh: float = 0.25
        m_shank: float = 0.25
        
        i_body: float = 1.0
        i_thigh: float = 0.25
        i_shank: float = 0.25

        body_len: float = 1.0
        thigh_len: float = 0.5
        shank_len: float = 0.5

    def __init__(self, q, v, c: Config = Config()):
        self.q = q
        self.v = v
        self.c = c

    @property
    def body_theta(self):
        return self.q[0]

    @property
    def body_x(self):
        return self.q[1]

    @property
    def body_y(self):
        return self.q[2]

    @property
    def left_hip_theta(self):
        return self.q[3]

    @property
    def left_knee_theta(self):
        return self.q[4]

    @property
    def right_hip_theta(self):
        return self.q[5]

    @property
    def right_knee_theta(self):
        return self.q[6]

    @property
    def body_com_location(self):
        return self.x[1:3]

    @property
    def body_com_to_hip_vec(self):
        return jnp.array([0, -self.c.body_len / 2])

    @property
    def hip_to_knee_vec(self):
        return jnp.array([0, -self.c.thigh_len])

    @property
    def knee_to_foot_vec(self):
        return jnp.array([0, -self.c.shank_len])

    @property
    def hip_location(self):
        return self.body_com_location + rot(self.body_theta) @ self.body_com_to_hip

    @property
    def left_knee_location(self):
        return (
            self.hip_location
            + rot(self.body_theta + self.left_hip_theta) @ self.hip_to_knee_vec
        )
    @property
    def right_knee_location(self):
        return (
            self.hip_location
            + rot(self.body_theta + self.right_hip_theta) @ self.hip_to_knee_vec
        )
    @property
    def left_thigh_com_location(self):
        return (
            self.hip_location
            + rot(self.body_theta + self.left_hip_theta) @ self.hip_to_knee_vec / 2
        )
    @property
    def right_thigh_com_location(self):
        return (
            self.hip_location
            + rot(self.body_theta + self.right_hip_theta) @ self.hip_to_knee_vec / 2
        )
    @property
    def left_foot_location(self):
        return (
            self.left_knee_location
            + rot(self.body_theta + self.left_hip_theta + self.left_knee_theta)
            @ self.knee_to_foot_vec
        )
    @property
    def right_foot_location(self):
        return (
            self.right_knee_location
            + rot(self.body_theta + self.right_hip_theta + self.right_knee_theta)
            @ self.knee_to_foot_vec
        )
    @property
    def left_shank_com_location(self):
        return (
            self.left_knee_location
            + rot(self.body_theta + self.left_hip_theta + self.left_knee_theta)
            @ self.knee_to_foot_vec
            / 2
        )
    @property
    def right_shank_com_location(self):
        return (
            self.right_knee_location
            + rot(self.body_theta + self.right_hip_theta + self.right_knee_theta)
            @ self.knee_to_foot_vec
            / 2
        )
        
    def ddt(self, f: Callable[["State"], jnp.ndarray]):

    def potential_energy(self):
        body_potential = self.c.m_body * self.c.g * self.body_y
        thigh_potential = (
            self.c.m_thigh
            * self.c.g
            * (self.left_thigh_com_location[1] + self.right_thigh_com_location[1])
        )
        shank_potential = (
            self.c.m_shank
            * self.c.g
            * (self.left_shank_com_location[1] + self.right_shank_com_location[1])
        )
            
        return body_potential + thigh_potential + shank_potential
        
    def kinetic_energy(self):
        linear_body_kinetic = 
            