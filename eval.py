import jax
import jax.numpy as jnp
from jax import flatten_util
import equinox as eqx
import tyro
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

from typing import Any, Literal
from jaxtyping import Float, Array, PRNGKeyArray, Integer, ScalarLike

from small_world.envs.from_map import FromMap
from small_world.envs.randcolors import RandColors
from small_world.envs.adventure import Adventure
from small_world.environment import Environment, EnvParams, Timestep

from main import Interaction, generate_population, build_rollout


def generate_gif(
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


def analysis(
    key: PRNGKeyArray,
    policy: eqx.Module,
    env: Environment,
    env_params: EnvParams,
    num_timesteps: int = 100,
    save_frames: bool = False,
):
    @eqx.filter_vmap
    def _forward_agents(k, pol, o, h):
        return pol(k, o, h, True)

    rollout_fn = jax.jit(
        build_rollout(env, env_params, num_timesteps, return_timestep=True)
    )
    timesteps = rollout_fn(key, policy, 0)
    frames = jax.vmap(env.render, in_axes=(None, 0, None))(env_params, timesteps, 20)
    np_frames = np.array(frames).astype(np.uint8)

    imgs = [Image.fromarray(frame) for frame in np_frames]

    fig = plt.figure()
    gs = GridSpec(4, 1, width_ratios=[2], height_ratios=[2, 1, 1, 1])
    axs = [fig.add_subplot(gs[i]) for i in range(gs.ncols * gs.nrows)]

    center = jnp.asarray((env_params.height / 2, env_params.width / 2))

    agent0_d, agent1_d, between_d = [], [], []
    entropies = [[], []]

    plt.rcParams.update({"font.size": 15})

    # Run a rollout of the policy and collect the interactions
    hstates = jnp.zeros(
        (
            env_params.num_agents,
            policy.in_layer.out_features,  # type: ignore[unresolved-attribute]
        )
    )

    if save_frames:
        fig2 = plt.figure(2)
        f2_ax = fig2.gca()

    for i, img in enumerate(imgs):
        timestep = jax.tree_util.tree_map(lambda x: x[i], timesteps)

        key, key_actions = jax.random.split(key)
        keys_actions = jax.random.split(key_actions, num=env_params.num_agents)
        _actions, hstates, logits = _forward_agents(
            keys_actions, policy, timestep.observations, hstates
        )

        probs = jax.nn.softmax(logits, axis=1)
        H = (-probs * jnp.log(probs)).sum(axis=1)
        entropies[0].append(H[0])
        entropies[1].append(H[1])

        agent_pos = timestep.state.agents_pos
        # distance to center
        centr_dists = jnp.sqrt(jnp.pow(agent_pos - center, 2.0).sum(axis=1))
        agent0_d.append(centr_dists[0])
        agent1_d.append(centr_dists[1])
        # distance between agents
        between_d.append(jnp.sqrt(jnp.pow(agent_pos[0] - agent_pos[1], 2.0).sum()))

        [ax.cla() for ax in axs]

        axs[0].imshow(img)
        axs[0].set_axis_off()

        # save frames if needed
        if save_frames:
            f2_ax.cla()
            f2_ax.imshow(img)
            f2_ax.set_axis_off()
            fig2.savefig(f"frames/frame_{i}.png")

        axs[1].cla()
        axs[1].plot(agent0_d, color="#7287fd")
        axs[1].plot(agent1_d, color="#e64553")
        axs[1].set_xlim(0, num_timesteps)
        axs[1].set_ylabel("Dist. to center")
        axs[1].set_xlabel("Timestep")

        axs[2].cla()
        axs[2].plot(between_d)
        axs[2].set_xlim(0, num_timesteps)
        axs[2].set_ylabel("Dist. between agents")
        axs[2].set_xlabel("Timestep")

        axs[3].cla()
        axs[3].plot(entropies[0], "b-")
        axs[3].plot(entropies[1], "r-")
        axs[3].set_xlim(0, num_timesteps)
        axs[3].set_ylim(0, -jnp.log(1 / env_params.num_actions).item())
        axs[3].set_ylabel("Entropy")
        axs[3].set_xlabel("Timestep")

        plt.pause(0.01)

    plt.show()


def main(
    seed: int = 42,
    params_file: str = "images/best_1.npy",
    env_id: Literal["from_map", "randcolors"] = "from_map",
    env_file: str = "./envs/empty_10x10.txt",
    view_size: int = 5,
    num_agents: int = 1,
    pol_hdim: int = 32,
    num_timesteps: int = 1024,
    pol_type: Literal["rnn", "mlp"] = "rnn",
    gif_file: str = "frames.gif",
    cmd: Literal["gif", "analysis"] = "analysis",
    save_frames: bool = False,
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
    agent = generate_population(
        key_pol, population_size=num_agents, use_rnn=pol_type == "rnn", kwpolicy=cfg_pol
    )

    par, _ = eqx.partition(agent, eqx.is_array)
    num_par = sum(x.size for x in jax.tree_util.tree_leaves(par))

    params = jnp.load(params_file)

    assert num_par == params.shape[0], (
        f"The number of parameters of the loaded file ({num_par}) and the policy generated from the config ({params.shape[0]}) do not match"
    )

    _, unravel_fn = flatten_util.ravel_pytree(agent)
    agent = unravel_fn(params)

    key_roll, key = jax.random.split(key)

    if cmd == "gif":
        generate_gif(key_roll, agent, env, env_params, num_timesteps, gif_file)

    elif cmd == "analysis":
        analysis(
            key=key_roll,
            policy=agent,
            env=env,
            env_params=env_params,
            num_timesteps=num_timesteps,
            save_frames=save_frames,
        )


if __name__ == "__main__":
    tyro.cli(main)
