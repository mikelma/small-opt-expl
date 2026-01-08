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
        hstate: Float[Array, " {self.rnn.hidden_size}"],
    ) -> tuple[Integer[ScalarLike, ""], Float[Array, " {self.rnn.hidden_size}"]]:
        x = obs.flatten()
        x = jax.nn.relu(self.in_layer(x))  # type: ignore[call-non-callable]

        # apply recurrent layer
        hstate = self.rnn(x, hstate)  # type: ignore[call-non-callable]
        x = hstate
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))

        logits = self.layers[-1](x)
        action = jax.random.categorical(key, logits)
        return action, hstate


class WorldModel(eqx.Module):
    layers: list

    def __init__(
        self,
        key: PRNGKeyArray,
        obs_dim: int,
        num_actions: int,
        seq_len: int,
        num_cells: int,
        hdim: int = 64,
        num_layers: int = 3,
    ):
        assert num_layers >= 2, (
            f"Minimum number of layers in WorldModel is 2 but got {num_layers}"
        )

        kemb, kl_in, kl_mid, kl_out = jax.random.split(key, 4)

        # input layer
        in_dim = seq_len * (obs_dim + num_actions)
        self.layers = [eqx.nn.Linear(in_dim, hdim, key=kl_in)]

        # intermediate layers
        for _ in range(num_layers - 2):
            kl_mid, _key = jax.random.split(kl_mid)
            self.layers.append(eqx.nn.Linear(hdim, hdim, key=_key))

        # output layer
        self.layers.append(eqx.nn.Linear(hdim, obs_dim * num_cells, key=kl_out))

    def __call__(
        self,
        obs: Float[Array, "seq_len view_size view_size"],
        one_hot_actions: Float[Array, " seq_len num_actions"],
    ) -> Float[Array, "view_size*view_size num_cells"]:
        flat_obs = obs.ravel()  # flatten the sequence of observations
        emb_act = one_hot_actions.ravel()

        x = jnp.hstack((flat_obs, emb_act))

        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))  # TODO use tanh?

        logits = self.layers[-1](x)
        logits = logits.reshape(obs.shape[1] * obs.shape[2], -1)

        return logits
