import os

os.environ["MUJOCO_GL"] = "egl"

import jax
from jax import numpy as jnp

import numpy as np

from dataclasses import dataclass

import mujoco

from training.env_config import EnvConfig

from jax.experimental.host_callback import id_tap
import wandb


@dataclass
class Finger:
    @classmethod
    def class_init(cls):
        cls.model = mujoco.MjModel.from_xml_path("assets/finger/scene.xml")
        cls.data = mujoco.MjData(cls.model)
        cls.home_state = jnp.array(cls.host_get_home_state())
        cls.renderer = mujoco.Renderer(cls.model, 512, 512)

    @classmethod
    def host_step(cls, state, action, env_config: EnvConfig):
        cls.model.opt.timestep = env_config.dt / env_config.substep
        cls.data.qpos[:] = state[: cls.model.nq]
        cls.data.qvel[:] = state[cls.model.nq :]

        for _ in range(env_config.substep):
            cls.data.ctrl[:] = action
            mujoco.mj_step(cls.model, cls.data)

        return cls.host_make_state()

    @classmethod
    def step(cls, state, action, env_config: EnvConfig):
        return jax.pure_callback(
            cls.host_step,
            state,
            *(state, action, env_config),
        )

    @classmethod
    def host_get_home_state(cls):
        mujoco.mj_resetData(cls.model, cls.data)

        return cls.host_make_state()

    @classmethod
    def get_home_state(cls):
        return cls.home_state

    @classmethod
    def host_make_state(cls):
        return np.concatenate([cls.data.qpos, cls.data.qvel], dtype=np.float32)

    @classmethod
    def get_config(cls):
        return EnvConfig(
            action_bounds=jnp.array(cls.model.actuator_ctrlrange),
            state_dim=cls.model.nq + cls.model.nv,
            act_dim=cls.model.nu,
            dt=cls.model.opt.timestep * 4,
            substep=4,
        )

    @classmethod
    def host_render_frame(cls, state):
        cls.data.qpos[:] = state[: cls.model.nq]
        cls.data.qvel[:] = state[cls.model.nq :]
        cls.renderer.update_scene(cls.data)
        img = cls.renderer.render()
        return img

    @classmethod
    def send_wandb_video_for_id_tap(cls, tap_pack, _):
        return # I can't fix this function, not sure why it's broken
        states, env_config = tap_pack
        fps = 24
        stride = int(1 / fps / env_config.dt)
        frames = [cls.host_render_frame(state) for state in states[::stride]]
        
        print(f"video shape: {np.stack(frames).shape}")

        wandb.log({"video": wandb.Video(np.stack(frames), fps=fps)})

    @classmethod
    def send_wandb_video(cls, states, env_config: EnvConfig):
        id_tap(
            cls.send_wandb_video_for_id_tap,
            (states, env_config),
        )


Finger.class_init()


# Testing code
if __name__ == "__main__":
    import time
    import mujoco.viewer

    with mujoco.viewer.launch_passive(Finger.model, Finger.data) as viewer:
        start = time.time()

        jax_state = Finger.get_home_state()
        jax_act = jnp.zeros(UnitreeGo1.model.nu)
        env_config = Finger.get_config()

        while viewer.is_running() and time.time() - start < 30:
            step_start = time.time()

            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.

            jax_state = Finger.step(jax_state, jax_act, env_config)

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = Finger.model.opt.timestep - (
                time.time() - step_start
            )

            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
