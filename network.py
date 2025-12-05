import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import PRNGKeyArray, Float, Array, Integer, ScalarLike


class RnnPolicy(eqx.Module):
    in_layer: eqx.Module
    rnn: eqx.Module
    layers: tuple

    def __init__(self, key: PRNGKeyArray, in_dim: int, out_dim: int, hdim: int = 64):
        kl0, krnn, kl1, kl2 = jax.random.split(key, 4)

        self.in_layer = eqx.nn.Linear(in_dim, hdim, key=kl0)

        self.rnn = eqx.nn.GRUCell(input_size=hdim, hidden_size=hdim, key=krnn)

        self.layers = (
            eqx.nn.Linear(hdim, hdim, key=kl1),
            eqx.nn.Linear(hdim, out_dim, key=kl2),
        )

    def __call__(
        self,
        key: PRNGKeyArray,
        obs: Float[Array, "view_size view_size"],
        hstate: Float[Array, "{self.rnn.hidden_size}"],
    ) -> tuple[Integer[ScalarLike, ""], Float[Array, "{self.rnn.hidden_size}"]]:
        x = obs.flatten()
        x = jax.nn.relu(self.in_layer(x))

        # apply recurrent layer
        hstate = self.rnn(x, hstate)

        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))

        logits = self.layers[-1](x)
        action = jax.random.categorical(key, logits)
        return action, hstate


class WorldModel(eqx.Module):
    layers: tuple
    act_emb: eqx.Module

    def __init__(
        self,
        key: PRNGKeyArray,
        obs_dim: int,
        num_actions: int,
        hdim: int = 64,
        act_emb_size: int = 16,
    ):
        kemb, kl0, kl1, kl2 = jax.random.split(key, 4)

        self.act_emb = eqx.nn.Embedding(
            num_embeddings=num_actions, embedding_size=act_emb_size, key=kemb
        )

        self.layers = (
            eqx.nn.Linear(obs_dim + act_emb_size, hdim, key=kl0),
            eqx.nn.Linear(hdim, hdim, key=kl1),
            eqx.nn.Linear(hdim, obs_dim, key=kl2),
        )

    def __call__(
        self,
        obs: Float[Array, "view_size view_size"],
        action: Integer[ScalarLike, ""],
    ) -> Float[Array, "view_size view_size"]:
        flat_obs = obs.ravel()  # flatten observations
        emb_act = self.act_emb(action)

        x = jnp.hstack((flat_obs, emb_act))

        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))  # TODO use tanh?

        pred = jax.nn.tanh(self.layers[-1](x))
        pred = pred.reshape(*obs.shape)
        return pred
