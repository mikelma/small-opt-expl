import jax
import jax.numpy as jnp
from jax import flatten_util
import equinox as eqx
import tyro
import numpy as np
from PIL import Image

from typing import Any, Literal
from jaxtyping import Float, Array, PRNGKeyArray, Integer, ScalarLike

from small_world.envs.from_map import FromMap
from small_world.envs.randcolors import RandColors
from small_world.envs.adventure import Adventure
from small_world.environment import Environment, EnvParams, Timestep

from main import Interaction, generate_population, build_rollout


def visualize_rollout(
    key: PRNGKeyArray,
    policy: eqx.Module,
    env: Environment,
    env_params: EnvParams,
    num_timesteps: int = 100,
    file_name: str = "frames.gif",
):
    rollout_fn = jax.jit(
        build_rollout(env, env_params, num_timesteps, return_timestep=True)
    )

    timesteps = rollout_fn(key, policy, 0)

    frames = jax.vmap(env.render, in_axes=(None, 0, None))(env_params, timesteps, 20)
    np_frames = np.array(frames).astype(np.uint8)

    imgs = [Image.fromarray(frame) for frame in np_frames]

    # duration is in milliseconds (50ms = 20fps)
    imgs[0].save(file_name, save_all=True, append_images=imgs[1:], duration=50, loop=0)


def main(
    seed: int = 42,
    params_file: str = "images/best_1.npy",
    env_id: Literal["from_map", "randcolors"] = "from_map",
    env_file: str = "./envs/empty_10x10.txt",
    view_size: int = 5,
    num_agents: int = 1,
    pol_hdim: int = 32,
):
    key = jax.random.key(seed)

    # configure and create the environment
    env_cfg: dict[str, Any] = dict(
        view_size=view_size,
        num_agents=num_agents,
    )
    if env_id == "from_map":
        env = FromMap()
        env_cfg["file_name"] = env_file

    elif env_id == "randcolors":
        env = RandColors()

    elif env_id == "adventure":
        env = Adventure()
        env_cfg["file_name"] = "./small-world/envs-txt/adventure_M.txt"
        env_cfg["agent_init_pos"] = jnp.asarray((17, 22))
        del env_cfg["num_agents"]
    else:
        raise Exception(f"Environment ID '{env_id}' not found")
    env_params = env.default_params(**env_cfg)

    key_pol, key = jax.random.split(key)
    cfg_pol = dict(
        in_dim=int(view_size**2),
        out_dim=env_params.num_actions,
        hdim=pol_hdim,
    )
    agent = generate_population(key_pol, population_size=num_agents, kwpolicy=cfg_pol)

    par, _ = eqx.partition(agent, eqx.is_array)
    num_par = sum(x.size for x in jax.tree_util.tree_leaves(par))

    params = jnp.load(params_file)

    assert num_par == params.shape[0], (
        f"The number of parameters of the loaded file ({num_par}) and the policy generated from the config ({params.shape[0]}) do not match"
    )

    _, unravel_fn = flatten_util.ravel_pytree(agent)
    agent = unravel_fn(params)

    key_roll, key = jax.random.split(key)
    visualize_rollout(key_roll, agent, env, env_params)


if __name__ == "__main__":
    tyro.cli(main)
