from finger import Finger

from physics.gen.mass_matrix import mass_matrix
from physics.gen.bias_forces import bias_forces

from physics.simulate import step
from physics.visualize import animate

import jax
import jax.numpy as jnp
import jax.profiler
import flax
from flax import linen as nn
from flax import struct
import optax

import jax.experimental.host_callback

from training.infos import Infos

import shutil

from einops import einops, einsum, reduce
import matplotlib.pyplot as plt

from embeds import EmbeddingLayer
from training.rollout import collect_rollout

from training.eval_actor import evaluate_actor
from training.vibe_state import (
    VibeState,
    TrainConfig,
)
from training.nets import (
    StateEncoder,
    ActionEncoder,
    StateSizer,
    TransitionModel,
    StateDecoder,
    ActionDecoder,
    encoded_state_dim,
    encoded_action_dim,
)

import orbax.checkpoint as ocp

# from unitree_go1 import UnitreeGo1

from training.train import train_step, dump_to_wandb
from policy import (
    random_policy,
    random_repeat_policy,
    make_target_conf_policy,
    make_finder_policy,
    make_piecewise_actor,
    random_action,
    PresetActor,
)  # , max_dist_policy

import timeit

import os

from contextlib import redirect_stdout

import wandb

seed = 1

# Generate random key
key = jax.random.PRNGKey(seed)
checkpoint_dir = "checkpoints"


checkpointer = ocp.PyTreeCheckpointer()

learning_rate = float(2.5e-4)
every_k = 1

env_cls = Finger

env_config = env_cls.get_config()

schedule = optax.cosine_onecycle_schedule(
    transition_steps=2048,
    peak_value=learning_rate,
    pct_start=0.125,
    div_factor=5.0,
    final_div_factor=5.0,
)

vibe_config = TrainConfig.init(
    learning_rate=learning_rate,
    optimizer=optax.MultiSteps(
        optax.chain(
            optax.zero_nans(),
            optax.clip_by_global_norm(1e12),
            optax.lion(learning_rate=learning_rate),
        ),
        every_k_schedule=every_k,
    ),
    state_encoder=StateEncoder(),
    action_encoder=ActionEncoder(),
    state_sizer=StateSizer(),
    transition_model=TransitionModel(1e4, 6, 64, 4),
    state_decoder=StateDecoder(env_config.state_dim),
    action_decoder=ActionDecoder(env_config.act_dim),
    env_config=env_config,
    seed=seed,
    rollouts=1024,
    epochs=64,
    batch_size=128,
    every_k=every_k,
    traj_per_rollout=256,
    rollout_length=64,
    state_radius=1.375,
    action_radius=2.0,
    reconstruction_weight=10.0,
    forward_weight=10.0,
    smoothness_weight=1.0,
    condensation_weight=1.0,
    dispersion_weight=1.0,
    forward_gate_sharpness=1024,
    smoothness_gate_sharpness=1024,
    dispersion_gate_sharpness=1,
    condensation_gate_sharpness=1,
    forward_gate_center=0,
    smoothness_gate_center=0,
    dispersion_gate_center=-9,
    condensation_gate_center=-9,
)

rng, key = jax.random.split(key)
vibe_state = VibeState.init(rng, vibe_config)

checkpoint_dir = "checkpoints"

# vibe_state = checkpointer.restore(
#     os.path.join(checkpoint_dir, "checkpoint_r38_s512.0"), item=vibe_state
# )

# clear checkpoints
if os.path.exists(checkpoint_dir):
    shutil.rmtree(checkpoint_dir)

os.makedirs(checkpoint_dir)

policy = jax.tree_util.Partial(random_policy)

start_state = env_cls.init()

rng, key = jax.random.split(key)
rngs = jax.random.split(rng, vibe_config.traj_per_rollout)

wandb.init(
    project="go_by_vibes",
    config={
        "pwd": os.getcwd(),
        **vibe_config.make_dict(),
    }
    # mode="disabled",
)


def do_rollout(carry_pack, _):
    (key, rollout_i, vibe_state) = carry_pack

    steps = vibe_config.epochs * vibe_config.traj_per_rollout / vibe_config.batch_size

    def checkpoint_for_id_tap(tap_pack, _):
        vibe_state, rollout_i, steps = tap_pack
        checkpoint_path = os.path.abspath(
            os.path.join(checkpoint_dir, f"checkpoint_r{rollout_i}_s{steps}")
        )
        checkpointer.save(checkpoint_path, vibe_state)

    jax.experimental.host_callback.id_tap(
        checkpoint_for_id_tap,
        (vibe_state, rollout_i, steps),
    )

    def collect_finder_rollout(key):
        rng, key = jax.random.split(key)
        target_state = (
            jax.random.ball(rng, d=encoded_state_dim, p=1)
            * vibe_config.state_radius
            * 2.0
        )
        rng, key = jax.random.split(key)
        actor, init_carry = make_finder_policy(
            rng,
            start_state,
            target_state,
            vibe_state,
            vibe_config,
            env_cls,
        )

        rng, key = jax.random.split(key)
        rollout_result = collect_rollout(
            start_state,
            actor,
            init_carry,
            env_cls,
            vibe_state,
            vibe_config,
            rng,
        )

        return rollout_result

    print("Collecting finder rollouts")
    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, vibe_config.traj_per_rollout)
    rollout_result = jax.vmap(collect_finder_rollout)(
        rngs,
    )

    video_states = jnp.nan_to_num(rollout_result[0][0])
    env_cls.host_send_wandb_video("Finder Video", video_states, env_config)

    def do_epoch(carry_pack, _):
        (
            key,
            epoch,
            vibe_state,
        ) = carry_pack

        # shuffle the data
        rng, key = jax.random.split(key)
        # Make a shuffled list of indices
        indices = jax.random.permutation(rng, rollout_result[0].shape[0])
        states = rollout_result[0][indices]
        actions = rollout_result[1][indices]

        jax.experimental.host_callback.id_tap(
            lambda epoch, _: print(f"Epoch {epoch}"), epoch
        )

        def process_batch(carry, rollout_result_batch):
            (
                key,
                chunk_i,
                vibe_state,
            ) = carry

            rollout_result_batch = (
                rollout_result_batch[0],
                rollout_result_batch[1][..., :-1, :],
            )

            rng, key = jax.random.split(key)
            (
                vibe_state,
                loss_infos,
            ) = train_step(
                rng,
                vibe_state,
                vibe_config,
                rollout_result_batch,
            )

            loss_infos.dump_to_console()

            msg = None

            jax.experimental.host_callback.id_tap(
                lambda chunk, _: print(f"Chunk {chunk}"), chunk_i
            )

            is_update_chunk = chunk_i % vibe_config.every_k == 0

            jax.lax.cond(is_update_chunk, loss_infos.dump_to_wandb, lambda: None)

            return (key, chunk_i + 1, vibe_state), (msg, loss_infos)

        states_batched = einops.rearrange(
            states, "(r b) t d -> r b t d", b=(vibe_config.batch_size // every_k)
        )
        actions_batched = einops.rearrange(
            actions, "(r b) t d -> r b t d", b=(vibe_config.batch_size // every_k)
        )

        rollout_results_batched = (states_batched, actions_batched)

        rng, key = jax.random.split(key)
        init = (
            rng,
            0,
            vibe_state,
        )

        (
            _,
            _,
            vibe_state,
        ), (
            msg,
            loss_infos,
        ) = jax.lax.scan(process_batch, init, rollout_results_batched)

        return (key, epoch + 1, vibe_state), loss_infos

    actor_eval_stride = min(1024, vibe_config.epochs)

    def do_epoch_set(carry_pack, _):
        (
            key,
            epoch,
            vibe_state,
        ) = carry_pack

        if rollout_i % 4 == 0:
            # Eval the actor every n epochs
            print("Evaluating actor")
            rng, key = jax.random.split(key)
            rngs = jax.random.split(rng, 32)

            eval_actor_partial = jax.tree_util.Partial(
                evaluate_actor,
                start_state=start_state,
                env_cls=env_cls,
                vibe_state=vibe_state,
                vibe_config=vibe_config,
                big_steps=64,
                small_steps=64,
                big_post_steps=0,
                small_post_steps=0,
            )

            (eval_states, _), infos, _, _ = jax.vmap(eval_actor_partial)(rngs)

            env_cls.host_send_wandb_video("Actor Video", eval_states[0], env_config)

            infos.dump_to_wandb()
            infos.dump_to_console()

        rng, key = jax.random.split(key)
        init = (rng, epoch, vibe_state)
        (_, epoch, vibe_state), infos = jax.lax.scan(
            do_epoch, init, None, length=actor_eval_stride
        )

        return (key, epoch, vibe_state), infos

    rng, key = jax.random.split(key)
    init = (rng, 0, vibe_state)

    jax.experimental.host_callback.id_tap(
        lambda rollout, _: print(f"Rollout {rollout}"), rollout_i
    )

    # Switching to a for loop to try and speed up compilation
    carry = init
    epoch_sets = max((vibe_config.epochs // actor_eval_stride), 1)
    for i in range(epoch_sets):
        carry, _ = do_epoch_set(carry, None)

    (_, _, vibe_state) = carry

    # (_, _, vibe_state), infos = jax.lax.scan(
    #     do_epoch_set, init, None, length=(vibe_config.epochs // actor_eval_stride)
    # )

    # jax.experimental.host_callback.id_tap(dump_infos_for_tap, (infos, rollout_i))

    return (key, rollout_i + 1, vibe_state), _


rng, key = jax.random.split(key)
init = (rng, jnp.array(0), vibe_state)

# Replacing the scan with a for loop to try and speed up compilation
carry = init
for i in range(vibe_config.rollouts):
    carry, _ = do_rollout(carry, None)

# (_, _, vibe_state), _ = jax.lax.scan(
#     do_rollout, init, None, length=vibe_config.rollouts
# )


jax.debug.print("Done!")
