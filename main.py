import jax
import jax.numpy as jnp
import tyro
from dataclasses import dataclass
import equinox as eqx
from small_world.envs.from_map import FromMap
from small_world.environment import Environment, EnvParams
from flax import struct
from functools import partial

from typing import Any
from jaxtyping import (
    Float,
    Array,
    Integer,
    PRNGKeyArray,
    ScalarLike,
    install_import_hook,
)

with install_import_hook("network", "beartype.beartype"):
    from network import RnnPolicy

from eda import generate_eda_state, EdaConfig


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


@dataclass
class Args:
    env: EnvConfig
    eda: EdaConfig
    policy: PolicyConfig

    seed: int = 42


@struct.dataclass
class Interaction:
    observation: Float[Array, "view_size view_size"]
    next_observation: Float[Array, "view_size view_size"]
    action: Integer[ScalarLike, ""]
    position: Integer[Array, "2"]


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


def build_rollout(env: Environment, env_params: EnvParams, num_timesteps: int = 100):
    def _rollout(key: jax.Array, policy: RnnPolicy):
        def _step(carry, key_step):
            timestep, hstate = carry

            key_step, key_action = jax.random.split(key_step)
            observation = timestep.observations[0]
            action, hstate = policy(key_action, observation, hstate)
            actions = action.reshape(
                1,
            )
            timestep = env.step(
                key_step,
                env_params,
                timestep,
                actions,
            )

            interaction = Interaction(
                observation,
                action,
                timestep.observations[0],
                timestep.state.agents_pos[0],
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

    for it in range(args.eda.num_generations):
        key, key_rolls, key_fs = jax.random.split(key, num=3)

        keys_rolls = jax.random.split(key_rolls, args.eda.population_size)
        interactions = jax.vmap(rollout_fn)(keys_rolls, population)

        print(interactions.observation.shape)
        quit()
