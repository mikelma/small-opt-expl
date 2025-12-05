import jax
import jax.numpy as jnp
import tyro
from dataclasses import dataclass
import equinox as eqx
from small_world.envs.from_map import FromMap
from small_world.environment import Environment, EnvParams, Timestep
from flax import struct
from functools import partial
import optax
from typing import Optional, Callable

from typing import Any
from jaxtyping import (
    Float,
    Array,
    Integer,
    PRNGKeyArray,
    ScalarLike,
    install_import_hook,
    jaxtyped,
)
from beartype import beartype as typechecker

with install_import_hook("network", "beartype.beartype"):
    from network import RnnPolicy, WorldModel

from eda import generate_eda_state, EdaConfig, eda_sample


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

    num_iterations: int = 20
    """number of total updates in training"""

    eval_batch_size: int = 128
    """evaluation batch size"""

    learning_rate: float = 0.001


@dataclass
class Args:
    env: EnvConfig
    eda: EdaConfig
    policy: PolicyConfig
    wm: WorldModelConfig

    seed: int = 42


@jaxtyped(typechecker=typechecker)
@struct.dataclass
class Interaction:
    observation: Float[Array, "*batch view_size view_size"]
    next_observation: Float[Array, "*batch view_size view_size"]
    action: Integer[ScalarLike, "*batch"]
    position: Integer[Array, "*batch 2"]


def generate_population(
    key: PRNGKeyArray, population_size: int, kwpolicy: Any = dict()
):
    @eqx.filter_vmap
    def _make_pop(key):
        return RnnPolicy(key, **kwpolicy)

    keys = jax.random.split(key, population_size)
    return _make_pop(keys)


def convert_bfloat16(module: eqx.Module) -> eqx.Module:
    params, static = eqx.partition(module, eqx.is_array)
    params = jax.tree.map(lambda a: a.astype(jnp.bfloat16), params)
    return eqx.combine(params, static)


def build_rollout(
    env: Environment, env_params: EnvParams, num_timesteps: int = 100
) -> Callable:
    def _rollout(key: jax.Array, policy: RnnPolicy):
        def _step(
            carry: tuple[Timestep, Float[Array, "{policy.rnn.hidden_size}"]],
            key_step: PRNGKeyArray,
        ) -> tuple[
            list[Timestep, Float[Array, "{policy.rnn.hidden_size}"]], Interaction
        ]:
            timestep, hstate = carry

            key_step, key_action = jax.random.split(key_step)
            observation = timestep.observations[0]
            action, hstate = policy(key_action, observation, hstate)
            actions = jnp.asarray((action,))
            timestep = env.step(
                key_step,
                env_params,
                timestep,
                actions,
            )

            interaction = Interaction(
                observation=observation,
                next_observation=timestep.observations[0],
                action=action,
                position=timestep.state.agents_pos[0],
            )
            return [timestep, hstate], interaction

        key_reset, key_keys = jax.random.split(key)

        timestep = env.reset(env_params, key_reset)

        # Run a rollout of the policy and collect the interactions
        init_hstate = jnp.zeros(policy.rnn.hidden_size)
        step_keys = jax.random.split(key_keys, num_timesteps)
        _carry, interactions = jax.lax.scan(_step, [timestep, init_hstate], step_keys)

        return interactions

    return _rollout


@jaxtyped(typechecker=typechecker)
def random_batch_indices(
    key: PRNGKeyArray,
    batch_size: int,
    num_batches: int,
    dataset_len: int,
    sequential: bool = True,
) -> Integer[Array, "{num_batches} {batch_size}"]:
    """Example output for batch_size=8, num_bathes=3, dataset_len: 100, and sequential=True:
        [[10  5  4  4 10 25  1  7]  # indices in [0, 33)
         [26 28 46 21 23 33 64 58]  # indices in [0, 66)
         [30 76 59 20 18 54 57 79]] # indices in [0, 99)

    Example output for the same arguments except sequential=False:
        [[91 61 47 17 86 24 37 97]
         [22 22 96 35 51 87 12 70]
         [59 34 63 55 36 62 65 34]].
    """

    def batch_idx(batch_key, batch_id):
        upper = jax.lax.select(
            sequential,
            on_true=(batch_id + 1) * (dataset_len // num_batches),
            on_false=dataset_len,
        )

        return jax.random.randint(batch_key, (batch_size,), 0, upper)

    keys = jax.random.split(key, num_batches)
    batch_ids = jnp.arange(num_batches)
    return jax.vmap(batch_idx)(keys, batch_ids)


@eqx.filter_jit
@jaxtyped(typechecker=typechecker)
def compute_fitness(
    key: PRNGKeyArray,
    env_params: EnvParams,
    interactions: Interaction,
    wm_cfg: WorldModelConfig,
):
    @partial(jax.vmap, in_axes=(None, None, 0))
    def _batched_loss_fn(model, data, batch_idx):
        obs = data.observation[batch_idx]
        act = data.action[batch_idx]
        tgt = data.next_observation[batch_idx]
        pred = model(obs, act)

        loss = optax.losses.squared_error(pred, tgt)
        return loss

    def _avg_loss(model, data, batch_idx):
        losses = _batched_loss_fn(model, data, batch_idx)
        return losses.mean()

    def _train_batch(carry, batch_idx):
        model, optim_state = carry

        loss, grads = eqx.filter_value_and_grad(_avg_loss)(
            model, interactions, batch_idx
        )

        updates, opt_state = optim.update(
            grads, optim_state, eqx.filter(model, eqx.is_array)
        )

        model = eqx.apply_updates(model, updates)

        new_carry = (model, optim_state)
        return new_carry, loss

    def _train_eval_step(carry, batch_indices):
        model, optim_state, key = carry

        # Scan across num_batches. `batch_indices` shape: (num_batches, batch_size).
        train_out, losses = jax.lax.scan(
            _train_batch, (model, optim_state), batch_indices
        )
        model, optim_state = train_out

        eval_loss = losses.mean()  # TODO eval the model in the test data here
        print("\n*** EVAL IN TRAIN DATA ***\n")

        # key, key_batch = jax.random.split(key)
        # eval_idx = random_batch_indices(
        #     key_batch, wm_cfg.eval_batch_size, 1, eval_dataset.action.shape[0]
        # )[0]
        # eval_loss = _avg_loss(model, eval_dataset, eval_idx)

        carry = (model, optim_state, key)

        return carry, eval_loss

    key_wm, key_batches, key_eval = jax.random.split(key, 3)

    # Initialize the world model and optimizer
    model = WorldModel(
        key_wm,
        hdim=wm_cfg.hdim,
        obs_dim=env_params.view_size**2,
        num_actions=env_params.num_actions,
    )
    optim = optax.adamw(wm_cfg.learning_rate)
    optim_state = optim.init(eqx.filter(model, eqx.is_array))

    # Pre-compute training batch indices. shape: (iters, num_batches, batch_size)
    dataset_len = interactions.observation.shape[0]
    batches = random_batch_indices(
        key_batches,
        wm_cfg.batch_size * wm_cfg.num_batches,
        wm_cfg.num_iterations,
        dataset_len,
    )
    batches = batches.reshape(
        wm_cfg.num_iterations, wm_cfg.num_batches, wm_cfg.batch_size
    )

    # Run the world model train-eval loop
    carry = (model, optim_state, key_eval)
    _carry, eval_losses = jax.lax.scan(_train_eval_step, carry, batches)

    return eval_losses


if __name__ == "__main__":
    args = tyro.cli(Args)

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

    compute_fitness_fn = partial(compute_fitness, env_params=env_params, wm_cfg=args.wm)
    batched_compute_fs = jax.vmap(compute_fitness_fn)

    for it in range(args.eda.num_generations):
        key, key_rolls, key_fs, key_eda = jax.random.split(key, num=4)

        keys_rolls = jax.random.split(key_rolls, args.eda.population_size)
        interactions = jax.vmap(rollout_fn)(keys_rolls, population)

        print("*** TODO *** Implement repeated fitness calculations")
        keys_fs = jax.random.split(key_fs, num=args.eda.population_size)
        fitness = batched_compute_fs(key=keys_fs, interactions=interactions)
        avg_fitness = fitness.sum(-1)

        print(
            f"{it + 1}/{args.eda.num_generations} fitness:",
            fitness.mean(),
            ", min:",
            fitness.min(),
            ", max:",
            fitness.max(),
        )

        # Generate new solutions
        ranking = jnp.argsort(avg_fitness)
        eda_state, population = eda_sample(
            key_eda,
            eda_state,
            population,
            ranking,
            elite_ratio=args.eda.elite_ratio,
            learning_rate=args.eda.learning_rate,
        )
