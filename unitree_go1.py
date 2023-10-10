import jax
from jax import numpy as jnp

import numpy as np

from dataclasses import dataclass

import mujoco


@dataclass
class UnitreeGo1:
    @classmethod
    def class_init(cls):
        cls.model = mujoco.MjModel.from_xml_path("assets/unitree_go1/scene.xml")
        cls.data = mujoco.MjData(cls.model)

    @classmethod
    def step(cls, state, action):
        cls.model.data.qpos[:] = state[: cls.model.nq]
        cls.model.data.qvel[:] = qd
        cls.model.data.ctrl[:] = action

        mujoco.mj_step(cls.model, cls.data)

        return cls.make_state()

    @classmethod
    def get_home_state(cls):
        mujoco.mj_resetData(cls.model, cls.data)

        return cls.make_state()

    @classmethod
    def make_state(cls):
        result = jnp.zeros(cls.model.nq * 2)

        result = result.at[: cls.model.nq].set(cls.data.qpos)
        result = result.at[cls.model.nq :].set(cls.data.qvel)

        return result


UnitreeGo1.class_init()


if __name__ == "__main__":
    import time
    import mujoco.viewer

    with mujoco.viewer.launch_passive(UnitreeGo1.model, UnitreeGo1.data) as viewer:
        start = time.time()

        jax_state = UnitreeGo1.get_home_state()
        jax_act = jnp.zeros(UnitreeGo1.model.nu)

        while viewer.is_running() and time.time() - start < 30:
            step_start = time.time()

            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.

            jax_state = UnitreeGo1.step(jax_state, jax_act)

            # Example modification of a viewer option: toggle contact points every two seconds.
            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = UnitreeGo1.model.opt.timestep - (
                time.time() - step_start
            )

            if time_until_next_step > 0:
                time.sleep(time_until_next_step)