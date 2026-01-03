import jax
import jax.numpy as jnp
import tyro
import dataclasses
import equinox as eqx
from small_world.envs.from_map import FromMap
from small_world.envs.randcolors import RandColors
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
import time
from evosax.problems import Problem
from evosax.algorithms import Open_ES as ES

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

# persistent compilation cache
jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches",
    "xla_gpu_per_fusion_autotune_cache_dir",
)


OptimState: TypeAlias = Any


@dataclasses.dataclass
class EnvConfig:
    id: str = "from_map"
    """environment's identifier"""

    file_name: str = "./envs/simple.txt"
    """path to the environment map (used when env.id="from_map")"""

    num_timesteps: int = 512
    """length of the policy rollouts"""

    view_size: int = 5
    """size of agents' view grid"""


@dataclasses.dataclass
class PolicyConfig:
    hdim: int = 128
    """size of the hidden layers"""


@dataclasses.dataclass
class EsConfig:
    num_generations: int = 100

    population_size: int = 8

    num_fs_repes: int = 4

    std_init: float = 0.05

    std_decay: float = 0.2

    lr_init: float = 0.01

    lr_decay: float = 0.1


class WorldModelConfig(eqx.Module):
    hdim: int = 128
    """size of the hidden layers"""

    num_layers: int = 3
    """total number of layers (including in and out)"""

    num_iterations: int = 32
    """number of total updates in training"""

    eval_batch_size: int = 128
    """evaluation batch size"""

    learning_rate: float = 0.001

    seq_len: int = 8
    """number of observations to include as input"""


@dataclasses.dataclass
class Args:
    env: EnvConfig
    es: EsConfig
    policy: PolicyConfig
    wm: WorldModelConfig

    seed: int = 42

    wandb: bool = False

    wandb_project_name: str = "small-opt-expl"

    pertub_probs: list[float] = dataclasses.field(
        default_factory=lambda: [0.25, 0.5, 0.75, 1.0]
    )
    """probs. of taking a random action in each pertub. set"""

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


def build_rollout(
    env: Environment,
    env_params: EnvParams,
    num_timesteps: int = 100,
    return_timestep: bool = False,
) -> Callable:
    @eqx.filter_jit
    def _rollout(
        key: PRNGKeyArray, policy: RnnPolicy, rand_act_prob: float = 0.0
    ) -> Interaction | Timestep:
        def _step(
            carry: tuple[Timestep, Float[Array, "{policy.rnn.hidden_size}"]],
            key_step: PRNGKeyArray,
        ) -> tuple[
            tuple[Timestep, Float[Array, "{policy.rnn.hidden_size}"]],
            Interaction | Timestep,
        ]:
            timestep, hstate = carry

            key_env, key_action, key_cond, key_rand = jax.random.split(key_step, num=4)
            observation = timestep.observations[0]
            position = timestep.state.agents_pos[0]
            action, hstate = policy(key_action, observation, hstate)

            p = jax.random.uniform(key_cond)  # random value in the range [0, 1]
            action = jax.lax.cond(
                p <= rand_act_prob,
                # take a random action with unif. probability
                lambda: jax.random.randint(key_rand, (), 0, env_params.num_actions),
                # select policy's action
                lambda: action,
            )

            actions = jnp.expand_dims(action, axis=0)
            timestep = env.step(
                key_env,
                env_params,
                timestep,
                actions,
            )

            if not return_timestep:
                interaction = Interaction(
                    observation=observation,
                    next_observation=timestep.observations[0],
                    action=action,
                    position=position,
                )

                return (timestep, hstate), interaction
            else:
                return (timestep, hstate), timestep

        key_reset, key_keys = jax.random.split(key)

        timestep = env.reset(env_params, key_reset)

        # Run a rollout of the policy and collect the interactions
        init_hstate = jnp.zeros(
            policy.rnn.hidden_size,  # type: ignore[unresolved-attribute]
        )
        step_keys = jax.random.split(key_keys, num_timesteps)
        _carry, out = jax.lax.scan(_step, (timestep, init_hstate), step_keys)

        return out

    return _rollout


def make_windows(
    data: Interaction,
    seq_len: int,
) -> tuple[
    Float[Array, "T seq_len view view"],
    Float[Array, "T seq_len"],
    Float[Array, "T view view"],
]:
    len_data = data.observation.shape[0]

    pad_len = seq_len - 1

    # pad observations
    obs_shape = data.observation.shape[1:]
    zeros_obs = jnp.zeros((pad_len, *obs_shape))
    padded_obs = jnp.concatenate([zeros_obs, data.observation], axis=0)

    # pad actions
    zeros_act = jnp.zeros((pad_len,), dtype=int)
    padded_act = jnp.concatenate([zeros_act, data.action], axis=0)

    # create rolling window indices
    # dims: (T, seq_len) -> e.g. [[0,1,2], [1,2,3], ...]
    start_indices = jnp.arange(len_data)
    window_indices = start_indices[:, None] + jnp.arange(seq_len)[None, :]

    obs_windows = padded_obs[window_indices]
    act_windows = padded_act[window_indices]
    targets = data.next_observation

    return obs_windows, act_windows, targets


@jaxtyped(typechecker=typechecker)
def compute_ece(
    key: PRNGKeyArray,
    env_params: EnvParams,
    train_data: Interaction,
    eval_data: Interaction,
    wm_cfg: WorldModelConfig,
) -> tuple[Float[Array, " {wm_cfg.num_iterations}"], eqx.Module]:
    # number of timesteps of a policy rollout
    num_policy_timesteps = train_data.observation.shape[0]
    # compute the world model's training batch size
    assert num_policy_timesteps % wm_cfg.num_iterations == 0, (
        "Number of policy timesteps must be divisible by the number of world model train steps"
    )
    batch_size = num_policy_timesteps // wm_cfg.num_iterations

    train_obs, train_acts, train_tgts = make_windows(train_data, wm_cfg.seq_len)
    eval_obs, eval_acts, eval_tgts = make_windows(eval_data, wm_cfg.seq_len)

    @partial(jax.vmap, in_axes=(None, 0, 0, 0))
    def _batched_loss_fn(
        model: eqx.Module,
        obs_seq: Float[Array, "seq_len view view"],
        act_seq: Integer[Array, " seq_len"],
        target: Float[Array, "view view"],
    ) -> Scalar:
        one_hot_acts = jax.nn.one_hot(act_seq, env_params.num_actions, axis=-1)
        pred = model(obs_seq, one_hot_acts)  # type: ignore[non-subscriptable]
        loss = optax.losses.squared_error(pred, target)
        return jnp.mean(loss)

    def _avg_loss(
        model: eqx.Module,
        batch_idx: Integer[Array, " total_steps"],
        mask: Bool[Array, " total_steps"],
    ) -> Scalar:
        b_obs = train_obs[batch_idx]
        b_act = train_acts[batch_idx]
        b_tgt = train_tgts[batch_idx]

        losses = _batched_loss_fn(model, b_obs, b_act, b_tgt)
        losses = (losses * mask).sum() / mask.sum()
        return losses

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

        # training
        loss, grads = eqx.filter_value_and_grad(_avg_loss)(model, all_indices, mask)
        updates, optim_state = optim.update(
            grads, optim_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)

        # evaluation
        key, key_batch = jax.random.split(key)
        len_eval = eval_data.observation.shape[0]
        # data indices where to eval the world model
        eval_indices = jax.random.randint(
            key_batch, (wm_cfg.eval_batch_size,), 0, len_eval
        )

        # Look up pre-computed windows
        e_obs = eval_obs[eval_indices]
        e_act = eval_acts[eval_indices]
        e_tgt = eval_tgts[eval_indices]
        # actions to one-hot
        e_act = jax.nn.one_hot(e_act, env_params.num_actions, axis=-1)

        # Compute loss (reuse the batched function)
        eval_preds = jax.vmap(model)(e_obs, e_act)
        eval_loss = jnp.mean(optax.losses.squared_error(eval_preds, e_tgt))

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
        num_layers=wm_cfg.num_layers,
    )
    optim = optax.adamw(wm_cfg.learning_rate)
    optim_state = optim.init(eqx.filter(model, eqx.is_array))

    batch_start_idx = jnp.arange(0, num_policy_timesteps, step=batch_size)

    # Run the world model train-eval loop
    carry = (model, optim_state, key_eval)
    carry, eval_losses = jax.lax.scan(_train_eval_step, carry, batch_start_idx)
    learned_model = carry[0]

    return eval_losses, learned_model


@jaxtyped(typechecker=typechecker)
def compute_fitness_repetitions(
    key: PRNGKeyArray,
    population: RnnPolicy,
    env: Environment,
    env_params: EnvParams,
    pertub_probs: Float[Array, " num_pertub"],
    population_size: int,
    num_repetitions: int,
    rollout_steps: int,
    wm_cfg: WorldModelConfig,
) -> tuple[
    Float[
        Array,
        "{num_repetitions} {population_size} {wm_cfg.num_iterations}",
    ],
    Interaction,
    Interaction,
    eqx.Module,
]:
    rollout_fn = build_rollout(env, env_params, num_timesteps=rollout_steps)

    def _population_rollout(
        key: PRNGKeyArray, rand_act_prob: float, num_repes: int
    ) -> Interaction:
        @jax.vmap
        def _repeat_rollouts(key: PRNGKeyArray) -> Interaction:
            keys = jax.random.split(key, population_size)
            interactions = eqx.filter_vmap(rollout_fn, in_axes=(0, 0, None))(
                keys, population, rand_act_prob
            )
            return interactions

        keys_repes = jax.random.split(key, num_repes)
        interactions = _repeat_rollouts(keys_repes)
        return interactions

    def _repeated_population_eval(
        key: PRNGKeyArray, train_data: Interaction, eval_data: Interaction
    ) -> tuple[
        Float[Array, "{num_repetitions} {population_size} {wm_cfg.num_iterations}"],
        eqx.Module,
    ]:
        @jax.vmap
        def _population_eval(
            key_eval: PRNGKeyArray, train_data: Interaction
        ) -> tuple[
            Float[Array, "{population_size} {wm_cfg.num_iterations}"], eqx.Module
        ]:
            keys_fs = jax.random.split(key_eval, population_size)
            # evaluate the population based on the collected iterations
            batched_compute_fs = jax.vmap(compute_ece, in_axes=(0, None, 0, None, None))
            fitnesses, models = batched_compute_fs(
                keys_fs,
                env_params,
                train_data,
                eval_data,
                wm_cfg,
            )
            return fitnesses, models

        keys = jax.random.split(key, num_repetitions)
        fs, models = _population_eval(keys, train_data)
        return fs, models

    def _generate_eval_set(
        key: PRNGKeyArray, probs: Float[Array, " num_probs"]
    ) -> Interaction:
        keys = jax.random.split(key, num=probs.shape[0])
        batched_data = eqx.filter_vmap(_population_rollout, in_axes=(0, 0, None))(
            keys, probs, 1
        )
        # stack every interaction in a single dimension
        eval_data = jax.tree_util.tree_map(
            lambda x: x.reshape(-1, *x.shape[4:]), batched_data
        )
        return eval_data

    key_train, key_eval, key_fs = jax.random.split(key, num=3)

    train_data = _population_rollout(
        key_train, num_repes=num_repetitions, rand_act_prob=0.0
    )

    eval_data = _generate_eval_set(key_eval, pertub_probs)

    fitnesses, models = _repeated_population_eval(key_fs, train_data, eval_data)

    return fitnesses, train_data, eval_data, models


class EceProblem(Problem):
    def __init__(self, cfg: Args, env: Environment, env_params: EnvParams):
        self.env = env
        self.env_params = env_params
        self.cfg = cfg

        self.pertub_probs = jnp.asarray(args.pertub_probs, dtype=float)

        key = jax.random.key(0)
        self.kwpolicy = dict(
            in_dim=env_params.view_size * env_params.view_size,
            out_dim=env_params.num_actions,
            hdim=cfg.policy.hdim,
        )

        key = jax.random.key(0)
        population = generate_population(
            key, population_size=cfg.es.population_size, kwpolicy=self.kwpolicy
        )
        _, self.population_static = eqx.partition(population, eqx.is_array)

    @partial(jax.jit, static_argnames=("self",))
    def sample(self, key: jax.Array):
        """Sample a solution in the search space."""
        solution = generate_population(key, population_size=1, kwpolicy=self.kwpolicy)
        params, _static = eqx.partition(solution, eqx.is_array)
        params = jax.tree_util.tree_map(lambda x: x[0], params)
        return params

    @partial(jax.jit, static_argnames=("self",))
    def eval(self, key, solutions, state):
        population = eqx.combine(solutions, self.population_static)
        batched_losses, train_data, eval_data, models = compute_fitness_repetitions(
            key=key,
            env=self.env,
            env_params=self.env_params,
            pertub_probs=self.pertub_probs,
            population=population,
            num_repetitions=self.cfg.es.num_fs_repes,
            population_size=self.cfg.es.population_size,
            wm_cfg=self.cfg.wm,
            rollout_steps=self.cfg.env.num_timesteps,
        )
        fitness = batched_losses.sum(-1).mean(0)
        return (
            fitness,
            state,
            {
                "batched_losses": batched_losses,
                "train_data": train_data,
                "eval_data": eval_data,
                "models": models,
            },
        )


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
    positions: Integer[Array, "num_interactions 2"],
) -> Integer[Scalar, ""]:
    num_interactions = positions.shape[0]
    uniques_static = jnp.unique(
        positions,
        axis=0,
        size=num_interactions,
        fill_value=jnp.asarray((-1, -1)),
    )
    num_uniques = (uniques_static >= 0).all(axis=1).sum()
    return num_uniques


def visit_matrix(
    positions: Integer[Array, "num_interactions 2"], env_params: EnvParams
) -> Integer[Array, "{env_params.height} {env_params.width}"]:
    def _count(m, position):
        i, j = position[0], position[1]
        m = m.at[i, j].set(m[i, j] + 1)
        return m, 0

    mat = jnp.zeros((env_params.height, env_params.width), dtype=int)
    m, _ = jax.lax.scan(_count, mat, positions)
    return m


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


def model_error_matrix(
    models: eqx.Module,
    eval_data: Interaction,
    env_params: EnvParams,
    index: int,
    num_model: int = 0,
    seq_len: int = 8,
) -> Float[Array, "{env_params.height} {env_params.width}"]:
    @partial(jax.vmap, in_axes=(None, 0, 0, 0))
    def _batched_loss_fn(
        model: eqx.Module,
        obs_seq: Float[Array, "seq_len view view"],
        act_seq: Integer[Array, " seq_len"],
        target: Float[Array, "view view"],
    ) -> Scalar:
        one_hot_acts = jax.nn.one_hot(act_seq, env_params.num_actions, axis=-1)
        pred = model(obs_seq, one_hot_acts)  # type: ignore[non-subscriptable]
        loss = optax.losses.squared_error(pred, target)
        return jnp.mean(loss)

    # prepare eval data (make sequences)
    eval_obs, eval_acts, eval_tgts = make_windows(eval_data, seq_len)
    positions = eval_data.position
    # select the model to evaluate
    model = jax.tree_util.tree_map(lambda x: x[num_model, index], models)

    errors = _batched_loss_fn(model, eval_obs, eval_acts, eval_tgts)

    def _step(carry, index):
        mat, count = carry
        i, j = positions[index, 0], positions[index, 1]
        mat = mat.at[i, j].set(mat[i, j] + errors[index])
        count = count.at[i, j].set(count[i, j] + 1)
        return (mat, count), 0

    # sum the errors at each position and count the number of data
    # points per position in two matrices
    indices = jnp.arange(positions.shape[0])
    mat = jnp.zeros((env_params.height, env_params.width))
    count = jnp.zeros((env_params.height, env_params.width))
    carry, _ = jax.lax.scan(_step, (mat, count), indices)

    # compute the average error per position
    mat_sum, mat_count = carry
    mat_avg = mat_sum / mat_count
    # set the positions with no data points to nan for the sake of visualization
    mat_avg = jnp.where(mat_count == 0, jnp.nan, mat_avg)

    return mat_avg


def main(args: Args):
    if args.wandb:
        import wandb

        run_name = f"Env__{args.seed}__{int(time.time())}"
        wandb.init(
            project=args.wandb_project_name,
            config=vars(args),
            name=run_name,
        )

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    key = jax.random.key(args.seed)

    if args.env.id == "from_map":
        env = FromMap()
        env_params = env.default_params(
            file_name=args.env.file_name, view_size=args.env.view_size
        )

    elif args.env.id == "randcolors":
        env = RandColors()
        env_params = env.default_params(view_size=args.env.view_size)

    else:
        raise Exception(f"Environment ID '{args.env.id}' not found")

    key, key_prob, key_sol, key_es = jax.random.split(key, 4)
    problem = EceProblem(cfg=args, env=env, env_params=env_params)
    problem_state = problem.init(key_prob)

    solution = problem.sample(key_sol)
    lr_schedule = optax.exponential_decay(
        init_value=args.es.lr_init,
        transition_steps=args.es.num_generations,
        decay_rate=args.es.lr_decay,
    )
    std_schedule = optax.exponential_decay(
        init_value=args.es.std_init,
        transition_steps=args.es.num_generations,
        decay_rate=args.es.std_decay,
    )
    es = ES(
        population_size=args.es.population_size,
        solution=solution,
        std_schedule=std_schedule,
        optimizer=optax.adam(learning_rate=lr_schedule),
    )
    params = es.default_params  # Use default parameters
    state = es.init(key_es, solution, params)

    # Number of parameters of policy and world model networks
    print(
        "[*] Number of parameters of the policy network:",
        state.best_solution.shape[0],
    )
    dummy_wm = WorldModel(
        key,
        seq_len=args.wm.seq_len,
        hdim=args.wm.hdim,
        obs_dim=env_params.view_size**2,
        num_actions=env_params.num_actions,
        num_layers=args.wm.num_layers,
    )
    dummy_par, _ = eqx.partition(dummy_wm, eqx.is_array)
    wm_n_par = sum(x.size for x in jax.tree_util.tree_leaves(dummy_par))
    print("[*] Number of parameters of world models:", wm_n_par)

    fig1, ax1 = plt.subplots()
    ymax = None

    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    cbar3 = None

    # get the number of empty cells in the environment (used in metrics in the main loop below)
    key, reset_key = jax.random.split(key)
    dummy_timestep = env.reset(env_params, reset_key)
    mask = empty_cells_mask(dummy_timestep.state.grid)
    num_empty_cells = mask.sum()

    ask_fn = jax.jit(es.ask)
    tell_fn = jax.jit(es.tell)
    eval_fn = jax.jit(problem.eval)

    for it in range(args.es.num_generations):
        key, key_ask, key_eval, key_tell, key_viz = jax.random.split(key, 5)

        population, state = ask_fn(key_ask, state, params)

        start = time.time()

        # Evaluate the fitness of the population
        fitness, problem_state, info = eval_fn(key_eval, population, problem_state)
        batched_losses = info["batched_losses"]
        interactions = info["train_data"]

        fitness.block_until_ready()
        elapsed = time.time() - start
        print(f"   > Elapsed time evaluating policies: {round(elapsed, 3)}s\n")

        state, metrics = tell_fn(key_tell, population, fitness, state, params)

        ranking = jnp.argsort(fitness)

        print(
            f"{it + 1}/{args.es.num_generations} fitness:",
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

        # the positions in which the best agent has been
        # dims: (num_repes, num_interactions, 2)
        positions = interactions.position[:, ranking[0], :, :]

        if args.wandb:
            # compute the number of unique positions that the agent with the
            # best fitness visits (on average through repetitions)
            batch_unique_visits = jax.vmap(number_unique_visits)(positions)
            coverage_frac = (batch_unique_visits / num_empty_cells).mean()
            wandb.log({"best agents coverage frac": coverage_frac}, step=it)

        # --- plots --- #
        if (it + 1) % args.plot_interval == 0 or (it + 1) == args.es.num_generations:
            ymax = batched_losses.max() if ymax is None else ymax
            ax1.cla()
            plot_eval_losses(batched_losses, fitness, fig1, ax1, ymax)

            ax2.cla()
            visit_mat = jax.vmap(visit_matrix, in_axes=(0, None))(positions, env_params)
            ax2.set_title("Best agent's average visits per cell")
            ax2.imshow(visit_mat.mean(0))  # average across repetitions

            if args.show:
                fig1.canvas.draw()
                fig2.canvas.draw()
                plt.pause(1e-7)

            else:
                fig1.savefig(f"{args.save_dir}/losses_{it + 1}.png")
                fig2.savefig(f"{args.save_dir}/visits_{it + 1}.png")

        if (it + 1) % args.viz_rollout_interval == 0 or (
            it + 1
        ) == args.es.num_generations:
            visualize_rollout(
                key=key_viz,
                population=population,
                id=int(ranking[0]),
                env=env,
                env_params=env_params,
                num_timesteps=args.env.num_timesteps,
                file_name=f"{args.save_dir}/frames_{it + 1}.gif",
            )

            start = time.time()
            err_mat = model_error_matrix(
                models=info["models"],
                eval_data=info["eval_data"],
                env_params=env_params,
                index=int(ranking[0]),
                seq_len=args.wm.seq_len,
            )
            err_mat.block_until_ready()
            elapsed = round(time.time() - start, 3)
            print(f"   > Elapsed time computing error matrix: {elapsed}s")
            ax3.cla()
            im3 = ax3.imshow(err_mat, vmin=0, cmap="RdYlGn_r")
            ax3.set_title("Best agent model's avg error")
            if cbar3 is None:
                cbar3 = fig3.colorbar(im3, ax=ax3)
            else:
                cbar3.update_normal(im3)
            if args.show:
                fig3.canvas.draw()
                plt.pause(1e-7)
            else:
                fig3.savefig(f"{args.save_dir}/errors_{it + 1}.png")


if __name__ == "__main__":
    args = tyro.cli(Args)

    main(args)
