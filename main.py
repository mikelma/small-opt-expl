import jax
import jax.numpy as jnp
import tyro
from dataclasses import dataclass
import equinox as eqx
from small_world.envs.from_map import FromMap

from network import RnnPolicy

from typing import Any
from jaxtyping import Float, Array, Integer, PRNGKeyArray
from beartype import beartype as typechecker


@dataclass
class EnvConfig:
    file_name: str = "./simple_env.txt"


@dataclass
class EdaConfig:
    population_size: int = 8


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
