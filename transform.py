# from jax import numpy as jnp
# from jax import grad, jit, vmap
# from jax import random
# from jax import lax

# class SmallTransform:
#     rot: jnp.ndarray
#     pos: jnp.ndarray

#     def __init__(self, rot: jnp.ndarray, pos: jnp.ndarray):
#         if rot.shape != (3):
#             raise ValueError("Rotation vector must be 3")
#         if pos.shape != (3):
#             raise ValueError("Position vector must be 3")

#         self.rot = rot
#         self.pos = pos

#     def to_quat(self) -> QuatTransform:


# class BigTransform:
#     se3: jnp.ndarray

#     def __init__(self, se3: jnp.ndarray):
#         if se3.shape != (4, 4):
#             raise ValueError("SE3 must be 4x4")
#         self.se3 = se3
