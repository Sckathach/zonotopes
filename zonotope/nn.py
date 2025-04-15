from typing import Tuple

from jaxtyping import Float
from torch import Tensor

from zonotope.zonotope import Zonotope


def linear(z: Zonotope, weight: Float[Tensor, "out in"], bias: Tensor) -> Zonotope:
    return z.mul(weight, "... in, out in -> ... out") + bias


def attention(
    z: Zonotope,  # Float[Tensor, "batch posn d_model"],
    q_w: Float[Tensor, "n_heads d_model d_head"],
    q_b: Float[Tensor, "n_heads d_head"],
    k_w: Float[Tensor, "n_heads d_model d_head"],
    k_b: Float[Tensor, "n_heads d_head"],
    v_w: Float[Tensor, "n_heads d_model d_head"],
    v_b: Float[Tensor, "n_heads d_head"],
    shortformer_pos_embed: Float[Tensor, "batch posn d_model"],
) -> Tuple[Zonotope, Zonotope, Zonotope]:
    values = (
        z.mul(
            v_w, "batch seq d_model, n_heads d_model d_head -> batch seq n_heads d_head"
        )
        + v_b
    )

    z_with_pos = z + shortformer_pos_embed

    queries = (
        z_with_pos.mul(
            q_w, "batch seq d_model, n_heads d_model d_head -> batch seq n_heads d_head"
        )
        + q_b
    )
    keys = (
        z_with_pos.mul(
            k_w, "batch seq d_model, n_heads d_model d_head -> batch seq n_heads d_head"
        )
        + k_b
    )

    return queries, keys, values
