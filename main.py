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

from einops import einops, einsum
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
    make_piecewise_actor,
    random_action,
)  # , max_dist_policy

import timeit

import os

from contextlib import redirect_stdout

import wandb

seed = 1

# Generate random key
key = jax.random.PRNGKey(seed)

### Set up physics sim stuff
rng, key = jax.random.split(key)

mass_matrix = jax.jit(mass_matrix)
bias_forces = jax.jit(bias_forces)

mass_config = jnp.array([1.0, 0.25, 0.25, 0.04, 0.01, 0.01])
shape_config = jnp.array([1.0, 0.25, 0.25])

rng, key = jax.random.split(key)
home_q = jnp.array([0, 0, -0.0, 0.5, -0.5, -0.5, 0.5], dtype=jnp.float32)
start_q = home_q + jax.random.uniform(rng, shape=(7,), minval=-0.4, maxval=0.4)
qd = jnp.array([0, 0, 0, 0, 0, 0, 0], dtype=jnp.float32)

### Set up RL stuff

checkpoint_dir = "checkpoints"

# clear checkpoints
if os.path.exists(checkpoint_dir):
    shutil.rmtree(checkpoint_dir)

os.makedirs(checkpoint_dir)

checkpointer = ocp.PyTreeCheckpointer()

learning_rate = float(1.0e-4)
every_k = 1

env_cls = Finger

env_config = env_cls.get_config()

vibe_config = TrainConfig.init(
    learning_rate=learning_rate,
    optimizer=optax.MultiSteps(
        optax.chain(
            optax.zero_nans(),
            optax.clip_by_global_norm(200.0),
            optax.lion(
                learning_rate=optax.cosine_onecycle_schedule(
                    4096,
                    peak_value=learning_rate,
                    pct_start=0.1,
                    div_factor=2.5,
                    final_div_factor=10.0,
                )
            ),
        ),
        every_k_schedule=every_k,
    ),
    state_encoder=StateEncoder(),
    action_encoder=ActionEncoder(),
    transition_model=TransitionModel(1e4, 6, 64, 4),
    state_decoder=StateDecoder(env_config.state_dim),
    action_decoder=ActionDecoder(env_config.act_dim),
    env_config=env_config,
    seed=seed,
    rollouts=256,
    epochs=128,
    batch_size=32,
    every_k=every_k,
    traj_per_rollout=1024,
    rollout_length=512,
    reconstruction_weight=1.0,
    forward_weight=1.0,
    smoothness_weight=1.0,
    condensation_weight=1.0,
    dispersion_weight=1.0,
    inverse_reconstruction_gate_sharpness=1,
    inverse_forward_gate_sharpness=1,
    inverse_reconstruction_gate_center=-3,
    inverse_forward_gate_center=-5,
    forward_gate_sharpness=1,
    smoothness_gate_sharpness=1,
    dispersion_gate_sharpness=1,
    condensation_gate_sharpness=1,
    forward_gate_center=-2,
    smoothness_gate_center=-2,
    dispersion_gate_center=-2,
    condensation_gate_center=-2,
)

rng, key = jax.random.split(key)
vibe_state = VibeState.init(rng, vibe_config)

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

send_min_conf_video = env_cls.make_wandb_sender("min conf video")
send_random_video = env_cls.make_wandb_sender("random video")
send_rng_conf_video = env_cls.make_wandb_sender("rng conf video")

send_actor_video = env_cls.make_wandb_sender("actor video")


def dump_to_wandb_for_tap(tap_pack, _):
    infos, rollout_i, epoch_i, chunk_i = tap_pack
    dump_to_wandb(infos, rollout_i, epoch_i, chunk_i, vibe_config)


def do_rollout(carry_pack, _):
    (key, rollout_i, vibe_state) = carry_pack

    steps = vibe_config.epochs * vibe_config.traj_per_rollout / vibe_config.batch_size

    def checkpoint_for_id_tap(tap_pack, _):
        vibe_state, rollout_i, steps = tap_pack
        checkpoint_path = os.path.join(
            checkpoint_dir, f"checkpoint_r{rollout_i}_s{steps}"
        )
        checkpointer.save(checkpoint_path, vibe_state)

    jax.experimental.host_callback.id_tap(
        checkpoint_for_id_tap,
        (vibe_state, rollout_i, steps),
    )

    def collect_conf_rollout(key):
        rng, key = jax.random.split(key)
        actor = make_target_conf_policy(
            rng,
            start_state,
            vibe_state,
            vibe_config,
            env_cls,
        )

        rng, key = jax.random.split(key)
        rollout_result = collect_rollout(
            start_state,
            actor,
            None,
            env_cls,
            vibe_state,
            vibe_config,
            rng,
        )

        return rollout_result

    def collect_rng_conf_rollout(key):
        rng, key = jax.random.split(key)
        conf_actor = make_target_conf_policy(
            rng,
            start_state,
            vibe_state,
            vibe_config,
            env_cls,
        )

        rng_actor = random_repeat_policy

        actor = make_piecewise_actor(
            conf_actor, rng_actor, vibe_config.rollout_length // 2
        )

        rng1, rng2, key = jax.random.split(key, 3)
        rollout_result = collect_rollout(
            start_state,
            actor,
            random_action(rng1, vibe_config.env_config.action_bounds),
            env_cls,
            vibe_state,
            vibe_config,
            rng2,
        )

        return rollout_result

    def collect_rng_rollout(key):
        rng1, rng2, key = jax.random.split(key, 3)
        rollout_result = collect_rollout(
            start_state,
            random_repeat_policy,
            random_action(rng1, vibe_config.env_config.action_bounds),
            env_cls,
            vibe_state,
            vibe_config,
            rng2,
        )

        return rollout_result

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, vibe_config.traj_per_rollout // 4)
    conf_states, conf_actions = jax.vmap(collect_conf_rollout)(
        rngs,
    )

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, vibe_config.traj_per_rollout // 4)
    rng_states, rng_actions = jax.vmap(collect_rng_rollout)(
        rngs,
    )

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, vibe_config.traj_per_rollout // 2)
    rng_conf_states, rng_conf_actions = jax.vmap(collect_rng_conf_rollout)(
        rngs,
    )

    # make backup rollouts to swap for nans
    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, vibe_config.traj_per_rollout)
    bup_states, bup_actions = jax.vmap(collect_rng_rollout)(
        rngs,
    )

    states = jnp.concatenate([conf_states, rng_conf_states, rng_states], axis=0)
    actions = jnp.concatenate([conf_actions, rng_conf_actions, rng_actions], axis=0)

    traj_has_nan = jnp.logical_or(
        jnp.logical_or(
            jnp.any(jnp.isnan(states), axis=(-1, -2)),
            jnp.any(jnp.isnan(actions), axis=(-1, -2)),
        ),
        jnp.logical_or(
            jnp.any(jnp.abs(states) > 1e4, axis=(-1, -2)),
            jnp.any(jnp.abs(actions) > 1e4, axis=(-1, -2)),
        ),
    )

    info = Infos.init()
    info = info.add_plain_info("rollout traj nan portion", jnp.mean(traj_has_nan))

    states = jnp.where(traj_has_nan[..., None, None], bup_states, states)
    actions = jnp.where(traj_has_nan[..., None, None], bup_actions, actions)

    info.dump_to_wandb()

    rollout_result = (states, actions)

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, 32)
    (eval_states, _), infos = jax.vmap(
        evaluate_actor, in_axes=(0, None, None, None, None)
    )(
        rngs,
        start_state,
        env_cls,
        vibe_state,
        vibe_config,
    )

    rng, key = jax.random.split(key)
    random_traj = jax.random.choice(rng, eval_states, axis=0)

    send_actor_video(random_traj, env_config)

    infos.dump_to_wandb()
    infos.dump_to_console()

    send_random_video(jnp.nan_to_num(rng_states[0]), env_config)
    send_min_conf_video(jnp.nan_to_num(conf_states[0]), env_config)
    send_rng_conf_video(jnp.nan_to_num(rng_conf_states[0]), env_config)

    # from jax import config
    # config.update("jax_disable_jit", True)

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

            # jax.experimental.host_callback.id_tap(
            #     dump_to_wandb_for_tap, (loss_infos, rollout_i, epoch, chunk_i)
            # )

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

        # jax.profiler.save_device_memory_profile("memory.prof")

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, vibe_config.epochs)

    rng, key = jax.random.split(key)
    init = (rng, 0, vibe_state)

    jax.experimental.host_callback.id_tap(
        lambda rollout, _: print(f"Rollout {rollout}"), rollout_i
    )

    (_, _, vibe_state), infos = jax.lax.scan(
        do_epoch, init, None, length=vibe_config.epochs
    )

    # jax.experimental.host_callback.id_tap(dump_infos_for_tap, (infos, rollout_i))

    return (key, rollout_i + 1, vibe_state), _


rng, key = jax.random.split(key)
init = (rng, jnp.array(0), vibe_state)

(_, _, vibe_state), _ = jax.lax.scan(
    do_rollout, init, None, length=vibe_config.rollouts
)


jax.debug.print("Done!")

# for i in range(16):
#     ani = animate(rollout_result[0][i, ..., :7], shape_config=shape_config, dt=dt)
#     plt.show()

# pass
