import torch as t
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from zonotope.functional import dot_product, softmax
from zonotope.zonotope import Zonotope


class Attention(nn.Module):
    W_Q: Float[Tensor, "n_heads d_model d_head"]
    b_Q: Float[Tensor, "n_heads d_head"]
    W_K: Float[Tensor, "n_heads d_model d_head"]
    b_K: Float[Tensor, "n_heads d_head"]
    W_V: Float[Tensor, "n_heads d_model d_head"]
    b_V: Float[Tensor, "n_heads d_head"]
    W_O: Float[Tensor, "n_heads d_head d_model"]
    b_O: Float[Tensor, "n_heads d_model"]


def attention(
    z: Zonotope,  # Float[Tensor, "batch posn d_model"],
    attn: Attention,
    shortformer_pos_embed: Float[Tensor, "batch posn d_model"],
    return_pattern: bool = False,
) -> Zonotope:
    d_head = attn.W_Q.shape[-1]

    values = (
        z.einsum(
            attn.W_V,
            "batch seq d_model, n_heads d_model d_head -> batch seq n_heads d_head",
        )
        + attn.b_V
    )

    z_with_pos = z + shortformer_pos_embed

    queries = (
        z_with_pos.einsum(
            attn.W_Q,
            "batch seq d_model, n_heads d_model d_head -> batch seq n_heads d_head",
        )
        + attn.b_Q
    )
    keys = (
        z_with_pos.einsum(
            attn.W_K,
            "batch seq d_model, n_heads d_model d_head -> batch seq n_heads d_head",
        )
        + attn.b_K
    )

    attn_scores = dot_product(
        queries,
        keys,
        "batch posn_Q nheads d_head, batch posn_K nheads d_head -> batch nheads posn_Q posn_K",
    )

    all_ones = t.ones(
        attn_scores.W_C.size(-2), attn_scores.W_C.size(-1), device=attn_scores.device
    )
    mask = t.triu(all_ones, diagonal=1).bool()
    IGNORE = t.tensor(float("-inf"), dtype=t.float32, device="cuda")

    attn_scores.W_C[..., mask] = IGNORE
    attn_scores.W_Ei[..., mask, :] = 0
    attn_scores.W_Es[..., mask, :] = 0

    attn_pattern = softmax(attn_scores * (1 / d_head**0.5))

    if return_pattern:
        return attn_pattern

    z = dot_product(
        values,
        attn_pattern,
        "batch posn_K nheads d_head, batch nheads posn_Q posn_K -> batch posn_Q nheads d_head",
    )

    attn_out = (
        z.einsum(
            attn.W_O,
            "batch posn_Q nheads d_head, nheads d_head d_model -> batch posn_Q d_model",
        )
        + attn.b_O
    )

    return attn_out
