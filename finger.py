import os

os.environ["MUJOCO_GL"] = "EGL"

import jax
from jax import numpy as jnp
from jax.experimental.host_callback import id_tap

import numpy as np

from dataclasses import dataclass

import mujoco
from mujoco import mjx

from training.env_config import EnvConfig

# from jax.experimental.host_callback import id_tap
import wandb

import time


@dataclass
class Finger:
    @classmethod
    def class_init(cls):
        cls.host_model = mujoco.MjModel.from_xml_path("assets/finger/scene.xml")
        cls.model = mjx.device_put(cls.host_model)
        cls.renderer = mujoco.Renderer(cls.host_model, 512, 512)

    # @classmethod
    # def host_step(cls, state, action, env_config: EnvConfig):
    #     cls.model.opt.timestep = env_config.dt / env_config.substep
    #     data.qpos[:] = state[: cls.model.nq]
    #     data.qvel[:] = state[cls.model.nq :]

    #     for _ in range(env_config.substep):
    #         cls.data.ctrl[:] = action
    #         mujoco.mj_step(cls.model, cls.data)

    #     return cls.host_make_state()

    @classmethod
    def step(cls, state, action, env_config: EnvConfig = None):
        if env_config is None:
            env_config = cls.get_config()

        # Configure the model to the env_config
        substep_dt = env_config.dt / env_config.substep
        model = cls.model
        temp_opt = model.opt.replace(timestep=substep_dt)
        model = model.replace(opt=temp_opt)

        # Make the data
        data = mjx.make_data(model)

        # Filter out nans in the action
        nan_action_elems = jnp.isnan(action)
        ctrl = jnp.where(nan_action_elems, data.ctrl, action)

        # Set the model state
        qpos = state[: model.nq]
        qvel = state[model.nq :]

        data = data.replace(qpos=qpos, qvel=qvel, ctrl=ctrl)

        def scanf(data, _):
            data = data.replace(ctrl=ctrl)
            data = mjx.step(model, data)
            return data, _

        next_data, _ = jax.lax.scan(
            scanf,
            data,
            xs=None,
            length=env_config.substep,
        )
        next_qpos = next_data.qpos
        next_qvel = next_data.qvel

        return jnp.concatenate([next_qpos, next_qvel], dtype=jnp.float32)

    # @classmethod
    # def host_make_state(cls):
    #     return np.concatenate([cls.data.qpos, cls.data.qvel], dtype=np.float32)

    @classmethod
    def make_state(cls, data):
        return jnp.concatenate([data.qpos, data.qvel], dtype=jnp.float32)

    @classmethod
    def init(cls):
        data = mjx.make_data(cls.model)
        data = data.replace(qpos=jnp.array([0, 0, 0]))

        return cls.make_state(data)

    @classmethod
    def get_config(cls):
        return EnvConfig(
            action_bounds=jnp.array(cls.model.actuator_ctrlrange),
            state_dim=cls.model.nq + cls.model.nv,
            act_dim=cls.model.nu,
            dt=cls.model.opt.timestep * 32,
            substep=32,
        )

    @classmethod
    def host_render_frame(cls, state):
        host_data = mujoco.MjData(cls.host_model)

        nq = int(cls.host_model.nq)

        host_data.qpos[:] = state[:nq]
        host_data.qvel[:] = state[nq:]

        mujoco.mj_forward(cls.host_model, host_data)

        cls.renderer.update_scene(host_data, "topdown")
        img = cls.renderer.render()

        return img

    @classmethod
    def host_make_video(cls, states, env_config: EnvConfig, fps=24):
        stride = 1 / fps / env_config.dt

        # Approximate the fps
        if stride < 1:
            # If the stride is less than 1, then we will raise it to 1 and set the fps as high as possible
            stride = 1
            fps = 1 / env_config.dt
        else:
            # Otherwise, we will round the stride to the nearest integer and set the fps to that
            stride = int(stride)
            fps = 1 / env_config.dt / stride

        frames = []
        next_state_i = 0
        while next_state_i < states.shape[0]:
            frames.append(cls.host_render_frame(states[next_state_i]))
            next_state_i += stride

        vid_arr = np.stack(frames).transpose(0, 3, 1, 2)
        return vid_arr

    @classmethod
    def host_send_wandb_video(cls, name, states, env_config):
        print(f"Sending video {name}")

        fps = 24
        video_array = cls.host_make_video(states, env_config, fps)

        wandb.log({name: wandb.Video(video_array, fps=fps)})

        print(f"Sent video {name}")

    @classmethod
    def make_wandb_sender(cls, video_name="video"):
        def sender_for_id_tap(tap_pack, _):
            (states, env_config) = tap_pack

            cls.host_send_wandb_video(video_name, states, env_config)

        def sender(states, env_config: EnvConfig):
            # Modified to not use id_tap so I can use the EGL backend when not jit compiling
            sender_for_id_tap((states, env_config), None)
            # id_tap(sender_for_id_tap, (states, env_config))

        return sender


Finger.class_init()
