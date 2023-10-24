from finger import Finger

from typing import Dict, Tuple, Any

import jax
from jax import numpy as jp

from flax import struct

import mujoco
from mujoco import mjx

from brax import envs
from brax.envs import State, Env
from brax.base import Motion, Transform


class FingerEnv(Env):
    def __init__(
        self,
        target_wheel_pos=1.25,
        ctrl_cost_weight=0.01,
        **kwargs,
    ):
        physics_steps_per_control_step = 5
        kwargs["physics_steps_per_control_step"] = kwargs.get(
            "physics_steps_per_control_step", physics_steps_per_control_step
        )

        self.phys_state = Finger.init()
        
        self._target_wheel_pos = target_wheel_pos
        self._ctrl_cost_weight = ctrl_cost_weight

    def reset(self, rng: jp.ndarray) -> State:
        """Resets the environment to an initial state."""

        data = Finger.init()

        obs = data
        reward, done, zero = jp.zeros(3)
        metrics = {}
        return State(data, obs, reward, done, metrics)

    def step(self, state: State, action: jp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics."""
        data0 = state.pipeline_state
        data = Finger.step(data0, action)

        def cost_func(
            state,
            action,
            ctrl_cost_weight=self._ctrl_cost_weight,
            target_wheel_pos=self._target_wheel_pos,
        ):
            state_cost = jp.abs(state[0] - target_wheel_pos)
            action_cost = ctrl_cost_weight * jp.linalg.norm(action, ord=1)

            return state_cost + action_cost

        reward = -cost_func(data, action)
        done = 0.0

        return state.replace(pipeline_state=data, obs=data, reward=reward, done=done)

    @property
    def observation_size(self) -> int:
        """The size of the observation vector returned in step and reset."""
        return 6

    @property
    def action_size(self) -> int:
        """The size of the action vector expected by step."""
        return 2

    @property
    def backend(self) -> str:
        """The physics backend that this env was instantiated with."""
        return "mjx"


envs.register_environment("finger", FingerEnv)
