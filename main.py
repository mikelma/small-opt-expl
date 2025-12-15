import jax
import jax.numpy as jnp
import tyro
from dataclasses import dataclass
import equinox as eqx
from small_world.envs.from_map import FromMap
from small_world.envs.simple import Simple
from small_world.utils import empty_cells_mask
from small_world.environment import Environment, EnvParams, Timestep
from flax import struct
from functools import partial
import optax
from typing import Callable
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

from typing import Any, TypeAlias
from jaxtyping import (
    Float,
    Array,
    Integer,
    PRNGKeyArray,
    ScalarLike,
    Scalar,
    Bool,
    install_import_hook,
    jaxtyped,
)
from beartype import beartype as typechecker

with install_import_hook("network", "beartype.beartype"):
    from network import RnnPolicy, WorldModel

from eda import generate_eda_state, EdaConfig, eda_sample

# persistent compilation cache
jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches",
    "xla_gpu_per_fusion_autotune_cache_dir",
)


OptimState: TypeAlias = Any


@dataclass
class EnvConfig:
    file_name: str = "./simple_env.txt"
    num_timesteps: int = 512


@dataclass
class PolicyConfig:
    hdim: int = 128
    """size of the hidden layers"""

    f16: bool = False
    """Use bfloat16 for the weights"""


class WorldModelConfig(eqx.Module):
    hdim: int = 128
    """size of the hidden layers"""

    batch_size: int = 16
    """training batch size"""

    num_batches: int = 4
    """number of batches to train at each train iteration"""

    num_iterations: int = 32
    """number of total updates in training"""

    eval_batch_size: int = 128
    """evaluation batch size"""

    learning_rate: float = 0.001

    seq_len: int = 8
    """number of observations to include as input"""


@dataclass
class Args:
    env: EnvConfig
    eda: EdaConfig
    policy: PolicyConfig
    wm: WorldModelConfig

    seed: int = 42

    wandb: bool = False

    wandb_project_name: str = "small-opt-expl"

    save_dir: str = "images"
    """directory where images and GIFs will be saved"""

    show: bool = False
    """plots are shown to screen if true, otherwise plots are saved to `--save-dir`"""

    plot_interval: int = 1
    """interval in which plots (e.g., loss curves) are generated"""

    viz_rollout_interval: int = 10
    """interval in which a GIF of the best agent's rollout is created"""


@struct.dataclass
class Interaction:
    observation: Float[Array, "*batch view_size view_size"]
    next_observation: Float[Array, "*batch view_size view_size"]
    action: Integer[ScalarLike, "*batch"]
    position: Integer[Array, "*batch 2"]


def generate_population(
    key: PRNGKeyArray, population_size: int, kwpolicy: Any = dict()
) -> eqx.Module:
    @eqx.filter_vmap
    def _make_pop(key: PRNGKeyArray) -> eqx.Module:
        return RnnPolicy(key, **kwpolicy)

    keys = jax.random.split(key, population_size)
    return _make_pop(keys)


def convert_bfloat16(module: eqx.Module) -> eqx.Module:
    params, static = eqx.partition(module, eqx.is_array)
    params = jax.tree.map(lambda a: a.astype(jnp.bfloat16), params)
    return eqx.combine(params, static)


def build_rollout(
    env: Environment,
    env_params: EnvParams,
    num_timesteps: int = 100,
    return_timestep: bool = False,
) -> Callable:
    @eqx.filter_jit
    def _rollout(key: PRNGKeyArray, policy: RnnPolicy) -> Interaction | Timestep:
        def _step(
            carry: tuple[Timestep, Float[Array, "{policy.rnn.hidden_size}"]],
            key_step: PRNGKeyArray,
        ) -> tuple[
            tuple[Timestep, Float[Array, "{policy.rnn.hidden_size}"]],
            Interaction | Timestep,
        ]:
            timestep, hstate = carry

            key_step, key_action = jax.random.split(key_step)
            observation = timestep.observations[0]
            action, hstate = policy(key_action, observation, hstate)
            actions = jnp.expand_dims(action, axis=0)
            timestep = env.step(
                key_step,
                env_params,
                timestep,
                actions,
            )

            if not return_timestep:
                interaction = Interaction(
                    observation=observation,
                    next_observation=timestep.observations[0],
                    action=action,
                    position=timestep.state.agents_pos[0],
                )

                return (timestep, hstate), interaction
            else:
                return (timestep, hstate), timestep

        key_reset, key_keys = jax.random.split(key)

        timestep = env.reset(env_params, key_reset)

        # Run a rollout of the policy and collect the interactions
        init_hstate = jnp.zeros(policy.rnn.hidden_size)  # type: ignore[unresolved-attribute]
        step_keys = jax.random.split(key_keys, num_timesteps)
        _carry, out = jax.lax.scan(_step, (timestep, init_hstate), step_keys)

        return out

    return _rollout


@eqx.filter_jit
@jaxtyped(typechecker=typechecker)
def compute_fitness(
    key: PRNGKeyArray,
    env_params: EnvParams,
    train_data: Interaction,
    eval_data: Interaction,
    wm_cfg: WorldModelConfig,
):
    # number of timesteps of a policy rollout
    num_policy_timesteps = train_data.observation.shape[0]
    # compute the world model's training batch size
    assert num_policy_timesteps % wm_cfg.num_iterations == 0, (
        "Number of policy timesteps must be divisible by the number of world model train steps"
    )
    batch_size = num_policy_timesteps // wm_cfg.num_iterations

    @partial(jax.vmap, in_axes=(None, None, 0))
    @jaxtyped(typechecker=typechecker)
    def _batched_loss_fn(
        model: eqx.Module, data: Interaction, batch_idx: Integer[Scalar, ""]
    ) -> Scalar:
        # take the last M number of observations starting from `batch_idx`.
        seq_idx = batch_idx - jnp.arange(wm_cfg.seq_len, dtype=int)[::-1]
        # if `batch_idx` is lower than wm.seq_len then (i.e., there're no
        # prev steps), fill the missing obs and acts with zeros.
        obs_shape = data.observation.shape[1:]
        mask = jnp.broadcast_to(
            (seq_idx >= 0)[:, None, None], (wm_cfg.seq_len, *obs_shape)
        )
        obs = jnp.where(
            mask, data.observation[seq_idx], jnp.zeros((wm_cfg.seq_len, *obs_shape))
        )
        # same for actions
        act = jnp.where(
            seq_idx >= 0,
            data.action[seq_idx],  # type: ignore[non-subscriptable]
            jnp.zeros((wm_cfg.seq_len,), dtype=int),
        )
        tgt = data.next_observation[batch_idx]
        pred = model(obs, act)  # type: ignore[non-callable]

        loss = optax.losses.squared_error(pred, tgt)
        return jnp.array(loss).mean()

    @jaxtyped(typechecker=typechecker)
    def _avg_loss(
        model: eqx.Module,
        data: Interaction,
        batch_idx: Integer[Array, "total_steps"],
        mask: Bool[Array, "total_steps"],
    ) -> Scalar:
        losses = _batched_loss_fn(model, data, batch_idx)
        losses = (losses * mask).sum() / mask.sum()
        return losses

    @jaxtyped(typechecker=typechecker)
    def _train_eval_step(
        carry: tuple[eqx.Module, OptimState, PRNGKeyArray],
        batch_start_idx: Integer[Array, ""],
    ) -> tuple[tuple[eqx.Module, OptimState, PRNGKeyArray], Scalar]:
        model, optim_state, key = carry

        # full batch training with the data until timestep number batch_start_idx.
        # generate the mask of valid training instances according to batch_start_idx
        all_indices = jnp.arange(num_policy_timesteps)
        valid_limit = batch_start_idx + batch_size
        mask = all_indices < valid_limit

        loss, grads = eqx.filter_value_and_grad(_avg_loss)(
            model, train_data, all_indices, mask
        )
        updates, optim_state = optim.update(
            grads, optim_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)

        # evaluate the current world model in the eval data
        key, key_batch = jax.random.split(key)
        len_data = eval_data.action.shape[0]  # type: ignore[possibly-missing-attribute]
        eval_batch_idx = jax.random.randint(
            key_batch, (wm_cfg.eval_batch_size,), 0, len_data
        )

        # use the full eval batch for evaluation
        eval_mask = jnp.ones(wm_cfg.eval_batch_size, dtype=bool)
        eval_loss = _avg_loss(model, eval_data, eval_batch_idx, eval_mask)

        carry = (model, optim_state, key)

        return carry, eval_loss

    key_wm, key_batches, key_eval = jax.random.split(key, 3)

    # Initialize the world model and optimizer
    model = WorldModel(
        key_wm,
        seq_len=wm_cfg.seq_len,
        hdim=wm_cfg.hdim,
        obs_dim=env_params.view_size**2,
        num_actions=env_params.num_actions,
    )
    optim = optax.adamw(wm_cfg.learning_rate)
    optim_state = optim.init(eqx.filter(model, eqx.is_array))

    batch_start_idx = jnp.arange(0, num_policy_timesteps, step=batch_size)

    # Run the world model train-eval loop
    carry = (model, optim_state, key_eval)
    _carry, eval_losses = jax.lax.scan(_train_eval_step, carry, batch_start_idx)

    return eval_losses


@eqx.filter_jit
@jaxtyped(typechecker=typechecker)
def compute_fitness_repetitions(
    key: PRNGKeyArray,
    population: RnnPolicy,
    env_params: EnvParams,
    population_size: int,
    num_repetitions: int,
    wm_cfg: WorldModelConfig,
) -> tuple[
    Float[
        Array,
        "{num_repetitions} {population_size} {wm_cfg.num_iterations}",
    ],
    Interaction,
]:
    def _population_rollout(key: PRNGKeyArray) -> Interaction:
        @jax.vmap
        def _repeat_rollouts(key: PRNGKeyArray) -> Interaction:
            keys = jax.random.split(key, population_size)
            interactions = eqx.filter_vmap(rollout_fn)(keys, population)
            return interactions

        keys_repes = jax.random.split(key, num_repetitions)
        interactions = _repeat_rollouts(keys_repes)
        return interactions

    def _repeated_population_eval(
        key: PRNGKeyArray, interactions: Interaction
    ) -> Float[Array, "{num_repetitions} {population_size} {wm_cfg.num_iterations}"]:
        # flatten the `num_repetitions`, `population_size`, and
        # `args.env.num_timesteps` (first three dims of each array in `interactions`)
        eval_data = jax.tree_util.tree_map(
            lambda x: x.reshape(-1, *x.shape[3:]), interactions
        )

        @jax.vmap
        def _population_eval(
            key: PRNGKeyArray, interactions: Interaction
        ) -> Float[Array, "{population_size} {wm_cfg.num_iterations}"]:
            # evaluate the population based on the collected iterations
            keys_fs = jax.random.split(key, population_size)
            batched_compute_fs = jax.vmap(
                compute_fitness, in_axes=(0, None, 0, None, None)
            )
            fitnesses = batched_compute_fs(
                keys_fs,
                env_params,
                interactions,
                eval_data,
                wm_cfg,
            )
            return fitnesses

        keys = jax.random.split(key, num_repetitions)
        fs = _population_eval(keys, interactions)
        return fs

    key_roll, key_fs = jax.random.split(key)
    many_interactions = _population_rollout(key_roll)
    many_fs = _repeated_population_eval(key_fs, many_interactions)
    return many_fs, many_interactions


def plot_eval_losses(losses, fitness, fig, ax, ymax, cmap="viridis_r"):
    pop_size = losses.shape[1]
    norm_fs = fitness - fitness.min()
    norm_fs /= norm_fs.max()

    cmap = plt.get_cmap(cmap)
    colors = cmap(norm_fs)

    xx = jnp.arange(losses.shape[-1])
    for i in range(pop_size):
        loss_avg = losses[:, i, :].mean(0)
        loss_std = losses[:, i, :].std(0)
        ax.plot(xx, loss_avg, color=colors[i], label=str(round(fitness[i], 3)))
        ax.fill_between(
            xx, loss_avg - loss_std, loss_avg + loss_std, alpha=0.3, color=colors[i]
        )

    ax.set_ylim(0, ymax)


def number_unique_visits(
    positions: Integer[Array, "num_inter 2"], num_interactions: int = 512
) -> Integer[Scalar, ""]:
    uniques_static = jnp.unique(
        positions,
        axis=0,
        size=num_interactions,
        fill_value=jnp.asarray((-1, -1)),
    )
    num_uniques = (uniques_static >= 0).all(axis=1).sum()
    return num_uniques


def visualize_rollout(
    key: PRNGKeyArray,
    population: eqx.Module,
    id: int,
    env: Environment,
    env_params: EnvParams,
    num_timesteps: int = 100,
    file_name: str = "frames.gif",
):
    rollout_fn = jax.jit(
        build_rollout(env, env_params, num_timesteps, return_timestep=True)
    )

    params, static = eqx.partition(population, eqx.is_array)
    policy_params = jax.tree_util.tree_map(lambda x: x[id], params)
    policy = eqx.combine(policy_params, static)

    timesteps = rollout_fn(key, policy)

    frames = jax.vmap(env.render, in_axes=(None, 0, None))(env_params, timesteps, 20)
    np_frames = np.array(frames).astype(np.uint8)

    imgs = [Image.fromarray(frame) for frame in np_frames]

    # duration is in milliseconds (50ms = 20fps)
    imgs[0].save(file_name, save_all=True, append_images=imgs[1:], duration=50, loop=0)


if __name__ == "__main__":
    args = tyro.cli(Args)

    if args.wandb:
        import wandb
        import time

        run_name = f"Env__{args.seed}__{int(time.time())}"
        wandb.init(
            project=args.wandb_project_name,
            config=vars(args),
            name=run_name,
        )

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    key = jax.random.key(args.seed)

    env = FromMap()
    env_params = env.default_params(file_name=args.env.file_name)

    # Generate the population
    key, key_pop = jax.random.split(key)
    population = generate_population(
        key_pop,
        args.eda.population_size,
        kwpolicy=dict(
            in_dim=env_params.view_size * env_params.view_size,
            out_dim=env_params.num_actions,
            hdim=args.policy.hdim,
        ),
    )

    if args.policy.f16:
        population = convert_bfloat16(population)

    rollout_fn = build_rollout(env, env_params, num_timesteps=args.env.num_timesteps)

    eda_state = generate_eda_state(population)

    # Number of parameters of policy networks
    params, _ = eqx.partition(population, eqx.is_array)
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(
        "[*] Number of parameters of the policy network:",
        param_count // args.eda.population_size,
    )

    fig, ax = plt.subplots()
    ymax = None

    # get the number of empty cells in the environment (used in metrics in the main loop below)
    key, reset_key = jax.random.split(key)
    dummy_timestep = env.reset(env_params, reset_key)
    mask = empty_cells_mask(dummy_timestep.state.grid)
    num_empty_cells = mask.sum()

    for it in range(args.eda.num_generations):
        key, key_eval, key_eda, key_viz = jax.random.split(key, num=4)

        batched_losses, interactions = compute_fitness_repetitions(
            key=key_eval,
            env_params=env_params,
            population=population,
            num_repetitions=args.eda.num_fs_repes,
            population_size=args.eda.population_size,
            wm_cfg=args.wm,
        )
        fitness = batched_losses.sum(-1).mean(0)
        ranking = jnp.argsort(fitness)

        print(
            f"{it + 1}/{args.eda.num_generations} fitness:",
            fitness.mean(),
            ", min:",
            fitness.min(),
            ", max:",
            fitness.max(),
        )

        if args.wandb:
            wandb.log(
                {
                    "mean fs": fitness.mean(),
                    "max fs": fitness.max(),
                    "min fs": fitness.min(),
                },
                step=it,
            )

        if (it + 1) % args.plot_interval == 0 or (it + 1) == args.eda.num_generations:
            ymax = batched_losses.max() if ymax is None else ymax
            plt.cla()
            plot_eval_losses(batched_losses, fitness, fig, ax, ymax)
            if args.show:
                plt.pause(1e-7)
            else:
                plt.savefig(f"{args.save_dir}/losses_{it + 1}.png")

        if (it + 1) % args.viz_rollout_interval == 0 or (
            it + 1
        ) == args.eda.num_generations:
            visualize_rollout(
                key=key_viz,
                population=population,
                id=int(ranking[0]),
                env=env,
                env_params=env_params,
                num_timesteps=args.env.num_timesteps,
                file_name=f"{args.save_dir}/frames_{it + 1}.gif",
            )

        if args.wandb:
            # compute the number of unique positions that the agent with the
            # best fitness visits (on average through repetitions)
            positions = interactions.position[:, ranking[0], :, :]
            batch_unique_visits = jax.vmap(number_unique_visits, in_axes=(0, None))(
                positions, args.env.num_timesteps
            )
            coverage_frac = (batch_unique_visits / num_empty_cells).mean()
            wandb.log({"best agents coverage frac": coverage_frac}, step=it)

        # Generate new solutions
        eda_state, population = eda_sample(
            key_eda,
            eda_state,
            population,
            ranking,
            elite_ratio=args.eda.elite_ratio,
            learning_rate=args.eda.learning_rate,
        )
