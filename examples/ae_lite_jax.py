import jax
import jax.numpy as jnp
import jax.random as jr

import equinox as eqx

from functools import partial

import einops

from jaxtyping import Array, Float, Int, PyTree, PRNGKeyArray
from typing import Tuple

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MultiHeadAttention(eqx.Module):
    dim: int
    heads: int
    dim_head: int
    to_qkv: eqx.nn.Linear
    to_out: eqx.nn.Linear
    use_flash: bool = eqx.field(static=True)

    def __init__(self, dim: int, heads: int, dim_head: int, use_flash: bool, *, key: PRNGKeyArray):
        keys = jr.split(key, num=2)

        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.use_flash = use_flash

        inner_dim = heads * dim_head

        self.to_qkv = eqx.nn.Linear(dim, 3 * inner_dim, use_bias=False, key=keys[0])
        self.to_out = eqx.nn.Linear(inner_dim, dim, use_bias=False, key=keys[1])

    @eqx.filter_jit
    def __call__(self, x: Float[Array, "t e"]):
        """ Pretend input doesn't have the batch dimension
        """
        # Emit q, k, v from x
        q, k, v = einops.rearrange(
            jax.vmap(self.to_qkv)(x),
            't (three h d) -> three h t d',
            three=3,
            h=self.heads
        )

        # Scaled dot product
        if self.use_flash:
            v_attn = jax.nn.scaled_dot_product_attention(
                q, k, v,
                scale=1.0/jnp.sqrt(self.dim_head)
            )
        else:
            w = einops.einsum(
                q, k,
                'h t_q d, h t_k d -> h t_q t_k'
            )
            w /= self.dim_head**0.5
            w = jax.nn.softmax(w, axis=-1) # Each q over all k should be interpreted as prob distribution
            v_attn = einops.einsum(
                w, v,
                'h t_q t_k, h t_k d -> h t_q d'
            )

            # Merge heads
            v_merged = einops.rearrange(
                v_attn,
                'h t d -> t (h d)'
            )

        return jax.vmap(self.to_out)(v_merged)

    def run(self, x: Float[Array, "t e"]):
        """ Pretend input doesn't have the batch dimension
        """
        # Emit q, k, v from x
        q, k, v = einops.rearrange(
            jax.vmap(self.to_qkv)(x),
            't (three h d) -> three h t d',
            three=3,
            h=self.heads
        )

        # Scaled dot product
        w = einops.einsum(
            q, k,
            'h t_q d, h t_k d -> h t_q t_k'
        )
        w /= self.dim_head**0.5
        w = jax.nn.softmax(w, axis=-1) # Each q over all k should be interpreted as prob distribution
        v_attn = einops.einsum(
            w, v,
            'h t_q t_k, h t_k d -> h t_q d'
        )

        # Merge heads
        v_merged = einops.rearrange(
            v_attn,
            'h t d -> t (h d)'
        )

        return jax.vmap(self.to_out)(v_merged)
