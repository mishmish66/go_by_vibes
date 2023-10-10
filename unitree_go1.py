import jax
from jax import numpy as jnp

import numpy as np

from dataclasses import dataclass

import mujoco

@dataclass
class UnitreeGo1:
    model: any
    
    @classmethod
    def class_init(cls):
        cls.model = mujoco.MjModel.from_xml_path("assets/unitree_go1/scene.xml")
    
    # @classmethod
    # def init(cls, q, qd):
        
        
UnitreeGo1.class_init()