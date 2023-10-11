import jax
import jax.numpy as jnp
import jax.lax
from jax.tree_util import register_pytree_node_class

from jax.scipy.stats.multivariate_normal import pdf as multinorm_pdf
from jax.tree_util import register_pytree_node

import flax
from flax import linen as nn
from flax.training import train_state
from flax import struct

import optax

from einops import einsum, rearrange, reduce

from physics import step

from .env_config import EnvConfig

from .nets import (
    encoded_state_dim,
    encoded_action_dim,
    StateEncoder,
    ActionEncoder,
    TransitionModel,
    StateDecoder,
    ActionDecoder,
)

from .rollout import collect_rollout

from dataclasses import dataclass

import os

import wandb


@register_pytree_node_class
@dataclass(frozen=True)
class TrainConfig:
    learning_rate: any
    optimizer: any

    state_encoder: any
    action_encoder: any
    transition_model: any
    state_decoder: any
    action_decoder: any

    env_config: any

    rollouts: int
    epochs: int
    batch_size: int
    every_k: int
    traj_per_rollout: int
    rollout_length: int

    reconstruction_weight: any
    forward_weight: any
    smoothness_weight: any
    dispersion_weight: any
    condensation_weight: any

    @classmethod
    def init(
        cls,
        learning_rate,
        optimizer,
        state_encoder,
        action_encoder,
        transition_model,
        state_decoder,
        action_decoder,
        env_config,
        rollouts=1024,
        epochs=128,
        batch_size=256,
        every_k=1,
        traj_per_rollout=2048,
        rollout_length=250,
        reconstruction_weight=1.0,
        forward_weight=1.0,
        smoothness_weight=1.0,
        dispersion_weight=1.0,
        condensation_weight=1.0,
    ):
        return cls(
            learning_rate=learning_rate,
            optimizer=optimizer,
            state_encoder=state_encoder,
            action_encoder=action_encoder,
            transition_model=transition_model,
            state_decoder=state_decoder,
            action_decoder=action_decoder,
            env_config=env_config,
            rollouts=rollouts,
            epochs=epochs,
            batch_size=batch_size,
            every_k=every_k,
            traj_per_rollout=traj_per_rollout,
            rollout_length=rollout_length,
            reconstruction_weight=reconstruction_weight,
            forward_weight=forward_weight,
            smoothness_weight=smoothness_weight,
            dispersion_weight=dispersion_weight,
            condensation_weight=condensation_weight,
        )

    def make_dict(self):
        return {
            "learning_rate": self.learning_rate,
            "traj_per_rollout": self.traj_per_rollout,
            "rollouts": self.rollouts,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "every_k": self.every_k,
            "rollout_length": self.rollout_length,
            "reconstruction_weight": self.reconstruction_weight,
            "forward_weight": self.forward_weight,
            "smoothness_weight": self.smoothness_weight,
            "dispersion_weight": self.dispersion_weight,
            "condensation_weight": self.condensation_weight,
        }

    def tree_flatten(self):
        return [None], {
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer,
            "state_encoder": self.state_encoder,
            "action_encoder": self.action_encoder,
            "transition_model": self.transition_model,
            "state_decoder": self.state_decoder,
            "action_decoder": self.action_decoder,
            "env_config": self.env_config,
            "rollouts": self.rollouts,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "every_k": self.every_k,
            "traj_per_rollout": self.traj_per_rollout,
            "rollout_length": self.rollout_length,
            "reconstruction_weight": self.reconstruction_weight,
            "forward_weight": self.forward_weight,
            "smoothness_weight": self.smoothness_weight,
            "dispersion_weight": self.dispersion_weight,
            "condensation_weight": self.condensation_weight,
        }

    @classmethod
    def tree_unflatten(cls, aux, data):
        return cls.init(
            learning_rate=aux["learning_rate"],
            optimizer=aux["optimizer"],
            state_encoder=aux["state_encoder"],
            action_encoder=aux["action_encoder"],
            transition_model=aux["transition_model"],
            state_decoder=aux["state_decoder"],
            action_decoder=aux["action_decoder"],
            env_config=aux["env_config"],
            rollouts=aux["rollouts"],
            epochs=aux["epochs"],
            batch_size=aux["batch_size"],
            every_k=aux["every_k"],
            traj_per_rollout=aux["traj_per_rollout"],
            rollout_length=aux["rollout_length"],
            reconstruction_weight=aux["reconstruction_weight"],
            forward_weight=aux["forward_weight"],
            smoothness_weight=aux["smoothness_weight"],
            dispersion_weight=aux["dispersion_weight"],
            condensation_weight=aux["condensation_weight"],
        )


class VibeState(struct.PyTreeNode):
    step: int

    state_encoder_params: any
    action_encoder_params: any
    transition_model_params: any
    state_decoder_params: any
    action_decoder_params: any

    opt_state: any

    @classmethod
    def init(cls, key, train_config: TrainConfig):
        rng, key = jax.random.split(key)
        rngs = jax.random.split(rng, 5)

        state_encoder_params = train_config.state_encoder.init(
            rngs[0],
            jnp.ones(train_config.env_config.state_dim),
        )
        action_encoder_params = train_config.action_encoder.init(
            rngs[1],
            jnp.ones(train_config.env_config.act_dim),
            jnp.ones(encoded_state_dim),
        )
        transition_model_params = train_config.transition_model.init(
            rngs[2],
            jnp.ones([encoded_state_dim]),
            jnp.ones([16, encoded_action_dim]),
            jnp.ones(16),
            10,
        )
        state_decoder_params = train_config.state_decoder.init(
            rngs[3],
            jnp.ones(encoded_state_dim),
        )
        action_decoder_params = train_config.action_decoder.init(
            rngs[4],
            jnp.ones(encoded_action_dim),
            jnp.ones(encoded_state_dim),
        )

        temp_state = cls(
            step=0,
            state_encoder_params=state_encoder_params,
            action_encoder_params=action_encoder_params,
            transition_model_params=transition_model_params,
            state_decoder_params=state_decoder_params,
            action_decoder_params=action_decoder_params,
            opt_state=None,
        )

        opt_state = train_config.optimizer.init(temp_state.extract_params())

        return temp_state.replace(
            opt_state=opt_state,
        )

    def extract_params(self):
        return {
            "state_encoder_params": self.state_encoder_params,
            "action_encoder_params": self.action_encoder_params,
            "transition_model_params": self.transition_model_params,
            "state_decoder_params": self.state_decoder_params,
            "action_decoder_params": self.action_decoder_params,
        }

    def assign_dict(self, params_dict):
        return self.replace(
            state_encoder_params=params_dict["state_encoder_params"],
            action_encoder_params=params_dict["action_encoder_params"],
            transition_model_params=params_dict["transition_model_params"],
            state_decoder_params=params_dict["state_decoder_params"],
            action_decoder_params=params_dict["action_decoder_params"],
        )

    def apply_gradients(self, grads, train_config: TrainConfig):
        updates, new_opt_state = train_config.optimizer.update(
            grads,
            self.opt_state,
            self.extract_params(),
        )
        new_params = optax.apply_updates(self.extract_params(), updates)
        return self.replace(
            step=self.step + 1,
            opt_state=new_opt_state,
        ).assign_dict(new_params)
