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
        return jax.random.categorical(key, logits), hstate


class WorldModel(eqx.Module):
    tile_embed: eqx.Module
    color_embed: eqx.Module
    action_embed: eqx.Module
    layers: tuple
    square: int = eqx.field(static=True)

    def __init__(self, key, hdim=512, embed_dim=16, view_grid_size=5):
        kl1, kl2, kem1, kem2, kem3 = jax.random.split(key, 5)

        self.tile_embed = eqx.nn.Embedding(
            num_embeddings=NUM_TILES,
            embedding_size=embed_dim,
            key=kem1,
        )
        self.color_embed = eqx.nn.Embedding(
            num_embeddings=NUM_COLORS,
            embedding_size=embed_dim,
            key=kem2,
        )

        self.action_embed = eqx.nn.Embedding(
            num_embeddings=NUM_ACTIONS,
            embedding_size=embed_dim,
            key=kem3,
        )

        self.square = view_grid_size * view_grid_size
        in_dim = self.square * embed_dim * 2
        self.layers = (
            eqx.nn.Linear(in_dim + embed_dim, hdim, key=kl1),
            eqx.nn.Linear(
                hdim, self.square * NUM_TILES + self.square * NUM_COLORS, key=kl2
            ),
        )

    def __call__(self, obs, act):
        tiles, colors = obs[:, :, 0].ravel(), obs[:, :, 1].ravel()

        emb_tiles = jax.vmap(self.tile_embed)(tiles)
        emb_colors = jax.vmap(self.color_embed)(colors)

        emb_act = self.action_embed(act)

        x = jnp.hstack((emb_tiles.reshape(-1), emb_colors.reshape(-1), emb_act))

        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))

        logits = self.layers[-1](x)

        tile_logits = logits[: NUM_TILES * self.square].reshape(self.square, NUM_TILES)
        colors_logits = logits[-NUM_COLORS * self.square :].reshape(
            self.square, NUM_COLORS
        )

        return tile_logits, colors_logits
