import jax
from jax import flatten_util
import equinox as eqx
from network import RnnPolicy
import jax.numpy as jnp
from functools import partial
from flax import struct
from dataclasses import dataclass
from typing import TypeAlias

Policy: TypeAlias = RnnPolicy


@dataclass
class EdaConfig:
    num_generations: int = 100

    population_size: int = 8

    elite_ratio: float = 0.5
    """ratio of the solutions to consider elite"""

    learning_rate: float = 0.3
    """learning rate of the EDA (set to 1 to disable)"""


@struct.dataclass
class EdaState:
    mean: jax.Array
    std: jax.Array
    iteration: int


@jax.vmap
def flat_nn(policy: Policy):
    flat_params, _unravel_params = flatten_util.ravel_pytree(policy)
    return flat_params


def generate_eda_state(policies: Policy):
    params_tree, static = eqx.partition(policies, eqx.is_array)
    flat_params = flat_nn(params_tree)

    mean = flat_params.mean(0)
    std = flat_params.std(0)

    return EdaState(mean, std, 0)


@partial(jax.jit, static_argnames=("elite_ratio"))
def eda_sample(
    key: jax.Array,
    state: EdaState,
    policies: Policy,
    sorted_idx: jax.Array,
    elite_ratio: float = 0.5,
    learning_rate: float = 0.1,
):
    # filter the array leafs of the population pytree
    pop_params_tree, static = eqx.partition(policies, eqx.is_array)

    # just flatten a single model to get the `func_unravel` function
    single_tree = jax.tree_util.tree_map(lambda p: p[0], pop_params_tree)
    flat_tree, func_unravel = flatten_util.ravel_pytree(single_tree)

    # flatten population's parameters
    flat_params = flat_nn(pop_params_tree)

    pop_size, num_params = flat_params.shape
    num_elite = int(pop_size * elite_ratio)
    num_samples = pop_size - num_elite

    sorted_params = flat_params[sorted_idx]

    # first `num_elite` items are set to True, the rest to False
    mask_idx = jnp.arange(pop_size) < num_elite
    mask_pop = jnp.repeat(
        mask_idx[:, None], num_params, axis=1
    )  # dim: (pop_size, num_params)

    # replace with 0s the parameters of the solutions that aren't in the elite set
    masked_pop = jnp.where(mask_pop, sorted_params, jnp.zeros_like(flat_params))

    # compute the mean of the elite set. dim: (num_params,)
    mean = (1 / num_elite) * masked_pop.sum(axis=0)

    # compute the std of the elite set
    masked_sub = jnp.where(mask_pop, sorted_params - mean, jnp.zeros_like(flat_params))
    var = (1 / num_elite) * jnp.pow(masked_sub, 2).sum(axis=0)
    std = jnp.sqrt(var)

    # moving average of the mean and std
    mean = learning_rate * mean + (1 - learning_rate) * state.mean
    std = learning_rate * std + (1 - learning_rate) * state.std

    # sample new solutions
    z = jax.random.normal(key, (num_samples, num_params))
    samples = mean + std * z

    zeros_pad = jnp.zeros((num_elite, num_params))
    padded_samples = jnp.vstack((zeros_pad, samples))

    # replace bad solutions in the original population with the new samples while maintaining the elite set
    new_flat_pop = jnp.where(
        mask_pop,
        sorted_params,
        padded_samples,
    )

    # in-place update the policies with their new parameters
    new_tree = jax.vmap(func_unravel)(new_flat_pop)

    policies = eqx.combine(new_tree, static)

    # update EDA's state
    state = state.replace(
        mean=mean,
        std=std,
        iteration=state.iteration + 1,
    )

    return state, policies
