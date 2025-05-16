from typing import Callable

import torch as t
from jaxtyping import Float
from torch import Tensor

from zonotope.classical.z import Zonotope
from zonotope.hybrid_constrained.hcz import HCZ


def get_permutation_matrix(h: HCZ) -> Float[Tensor, "N N"]:
    r = h.zeros(h.N, h.N)
    for i in range(h.N // 2):
        r[2 * i, i] = 1
        r[-1 - 2 * i, -i - 1] = 1
    return r


def get_abstract_transformer_from_classical_zonotope(
    lower: Tensor | float,
    upper: Tensor | float,
    classical_zonotope_abstract_transformer: Callable[[Zonotope], Zonotope],
) -> HCZ:
    z = Zonotope.from_bounds(lower=t.as_tensor([lower]), upper=t.as_tensor([upper]))
    r = classical_zonotope_abstract_transformer(z)

    return HCZ.from_values(
        [z.W_C[0].item(), r.W_C[0].item()],
        [[z.W_Ei[0, 0].item(), 0], r.W_Ei.tolist()[0]],
    )


def abstract_relu(alpha: Tensor | float, beta: Tensor | float) -> HCZ:
    h1 = HCZ.from_values(
        center=[-alpha / 2, 0], continuous_generators=[[alpha / 2], [0]]
    )
    h2 = HCZ.from_values(
        center=[beta / 2, beta / 2], continuous_generators=[[beta / 2], [beta / 2]]
    )
    return h1.union(h2)


def abstract_from_classical(
    alpha: Tensor | float,
    beta: Tensor | float,
    classical_zonotope_abstract_transformer: Callable[[Zonotope], Zonotope],
) -> HCZ:
    c = (-alpha + beta) / 2
    h1 = get_abstract_transformer_from_classical_zonotope(
        -alpha, c, classical_zonotope_abstract_transformer
    )
    h2 = get_abstract_transformer_from_classical_zonotope(
        c, beta, classical_zonotope_abstract_transformer
    )
    return h1.union(h2)


def apply_abstract_transformer(
    z_in: HCZ,
    abs_transformer: Callable[[Tensor | float, Tensor | float], HCZ],
    **kwargs,
) -> HCZ:
    lower, upper = z_in.concretize(**kwargs)
    z_abs = HCZ.empty()

    for i in range(z_in.N):
        alpha, beta = -lower[i], upper[i]
        z_abs = z_abs.cartesian_product(abs_transformer(alpha, beta))

    perm = get_permutation_matrix(z_abs)
    r_in = t.cat([z_in.eye(z_in.N), z_in.zeros(z_in.N, z_in.N)], dim=-1)
    perm_z_abs = z_abs.einsum(perm, "Na, Nb Na -> Nb")
    intermediate_result = perm_z_abs.general_intersect(z_in, r_in)
    r_out = t.cat([z_in.zeros(z_in.N, z_in.N), z_in.eye(z_in.N)], dim=-1)
    result = intermediate_result.einsum(r_out, "Na, Nb Na -> Nb")
    return result


def dot_product(h1: HCZ, h2: HCZ, **kwargs) -> HCZ:
    a, b = h1.clone(), h2.clone()

    

    return a