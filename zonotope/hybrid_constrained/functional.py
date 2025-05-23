from functools import partial
from typing import Callable

import torch as t
from einops import einsum
from jaxtyping import Float
from torch import Tensor

from zonotope.classical.functional import exp as classical_exp
from zonotope.classical.functional import reciprocal as classical_reciprocal
from zonotope.classical.functional import tanh as classical_tanh
from zonotope.classical.z import Zonotope
from zonotope.hybrid_constrained.hcz_v2 import HCZ


def get_abstract_transformer_from_classical_zonotope(
    lower: Tensor | float,
    upper: Tensor | float,
    classical_zonotope_abstract_transformer: Callable[[Zonotope], Zonotope],
) -> HCZ:
    if t.allclose(t.as_tensor(lower), t.as_tensor(upper)):
        z = Zonotope.from_values(center=t.as_tensor([lower]))
    else:
        z = Zonotope.from_bounds(lower=t.as_tensor([lower]), upper=t.as_tensor([upper]))
    r = classical_zonotope_abstract_transformer(z)

    if z.Ei > 0 and r.Ei > 0:
        return HCZ.from_values(
            [z.W_C[0].item(), r.W_C[0].item()],
            [[z.W_Ei[0, 0].item(), 0], r.W_Ei.tolist()[0]],
        )

    return HCZ.from_values(
        [z.W_C[0].item(), r.W_C[0].item()],
    )


def abstract_relu(lower: Tensor | float, upper: Tensor | float) -> HCZ:
    if lower >= 0:
        return HCZ.from_values(
            [(upper + lower) / 2, (upper + lower) / 2],
            [[(upper - lower) / 2], [(upper - lower) / 2]],
        )
    if upper <= 0:
        return HCZ.from_values([(upper + lower) / 2, 0], [[(upper - lower) / 2], [0]])

    h1 = HCZ.from_values([lower / 2, 0], [[lower / 2], [0]])
    h2 = HCZ.from_values([upper / 2, upper / 2], [[upper / 2], [upper / 2]])
    return h1.union(h2)


def relu(z: HCZ, **kwargs) -> HCZ:
    return apply_abstract_transformer(z, abstract_relu, **kwargs)


def reciprocal(z: HCZ, **kwargs) -> HCZ:
    fn = partial(
        get_abstract_transformer_from_classical_zonotope,
        classical_zonotope_abstract_transformer=classical_reciprocal,
    )

    return apply_abstract_transformer(z, fn, **kwargs)


def exp(z: HCZ, **kwargs) -> HCZ:
    fn = partial(
        get_abstract_transformer_from_classical_zonotope,
        classical_zonotope_abstract_transformer=classical_exp,
    )

    return apply_abstract_transformer(z, fn, **kwargs)


def tanh(z: HCZ, **kwargs) -> HCZ:
    fn = partial(
        get_abstract_transformer_from_classical_zonotope,
        classical_zonotope_abstract_transformer=classical_tanh,
    )

    return apply_abstract_transformer(z, fn, **kwargs)


def abstract_from_classical(
    lower: Tensor | float,
    upper: Tensor | float,
    classical_zonotope_abstract_transformer: Callable[[Zonotope], Zonotope],
) -> HCZ:
    c = (lower + upper) / 2
    h1 = get_abstract_transformer_from_classical_zonotope(
        lower, c, classical_zonotope_abstract_transformer
    )
    h2 = get_abstract_transformer_from_classical_zonotope(
        c, upper, classical_zonotope_abstract_transformer
    )
    return h1.union(h2)


def apply_abstract_transformer(
    z_in: HCZ,
    abs_transformer: Callable[[Tensor | float, Tensor | float], HCZ],
    **kwargs,
) -> HCZ:
    lower, upper = z_in.concretize(**kwargs)
    lower_flat, upper_flat = lower.reshape(-1), upper.reshape(-1)
    z_abs = HCZ.empty()

    for i in range(z_in.N):
        z_abs = z_abs.cartesian_product(abs_transformer(lower_flat[i], upper_flat[i]))

    z_abs = z_abs.to(device=z_in.device, dtype=z_in.dtype)
    perm_z_abs = z_abs.rearrange("(x y) -> (y x)", x=z_in.N)
    r_in = z_in.cat([z_in.eye(z_in.N), (z_in.N, z_in.N)])
    intermediate_result = perm_z_abs.intersect(z_in, r_in)
    r_out = z_in.cat([(z_in.N, z_in.N), z_in.eye(z_in.N)])
    result = intermediate_result.einsum(r_out, "Na, Nb Na -> Nb")
    return result.reshape(*z_in.shape)


def dot_product(a: HCZ, b: HCZ, pattern: str, **kwargs) -> HCZ:
    h1, h2 = a.clone(), b.clone()

    dims_a, dims_bc = pattern.split(",")
    dims_b, dims_c = dims_bc.split("->")

    l1, u1 = h1.concretize(**kwargs)
    l2, u2 = h2.concretize(**kwargs)
    lower = einsum(l1 - h1.W_C, l2 - h2.W_C, pattern)
    upper = einsum(u1 - h1.W_C, u2 - h2.W_C, pattern)

    return h1.clone(
        W_c=einsum(h1.W_C, h2.W_C, pattern),
        W_G=h1.cat(
            [
                einsum(h2.W_C, h1.W_G, f"{dims_b}, {dims_a} I1 -> {dims_c} I1"),
                einsum(h1.W_C, h2.W_G, f"{dims_a}, {dims_b} I2 -> {dims_c} I2"),
                lower.unsqueeze(-1),
                (*h1.shape, 1),
                upper.unsqueeze(-1),
                (*h1.shape, 1),
            ]
        ),
        W_Gp=h1.cat(
            [
                einsum(h2.W_C, h1.W_Gp, f"{dims_b}, {dims_a} Ip1 -> {dims_c} Ip1"),
                einsum(h1.W_C, h2.W_Gp, f"{dims_a}, {dims_b} Ip2 -> {dims_c} Ip2"),
            ],
        ),
        W_A=h1.cat(
            [h1.W_A, (h1.J, h2.I + 4)],
            [(h2.J, h1.I), h2.W_A, (h2.J, 4)],
            [(2, h1.I + h2.I), h1.as_tensor([[1, 1, 0, 0], [0, 0, 1, 1]])],
        ),
        W_Ap=h1.cat(
            [h1.W_Ap, (h1.J, h2.Ip)], [(h2.J, h1.Ip), h2.W_Ap], [(2, h1.Ip + h2.Ip)]
        ),
        W_b=h1.cat([h1.W_B, h2.W_B, h1.as_tensor([-1, 1])]),
    )


def linear(z: HCZ, weight: Float[Tensor, "out in"], bias: Tensor) -> HCZ:
    return z.einsum(weight, "... in, out in -> ... out") + bias


def replace_invalid_values(
    z: HCZ, replace_value: float, reset_error_terms: bool = True
) -> HCZ:
    result = z.clone()

    mask = (z.W_C != z.W_C) | (float("inf") == z.W_C) | (float("-inf") == z.W_C)
    result.W_C[mask] = replace_value

    if reset_error_terms:
        result.W_G[mask, :] = 0
        result.W_Gp[mask, :] = 0

    return result


def softmax(z: HCZ, masked: bool = True, **kwargs) -> HCZ:
    a = z.repeat("... -> ... N", N=z.W_C.shape[-1])
    b = a.rearrange("... Ni Nj -> ... Nj Ni")
    z_diff = a - b

    if masked:
        z_diff = replace_invalid_values(z_diff, replace_value=float("-inf"))

    z_exp = exp(z_diff, **kwargs)
    z_sum = z_exp.sum(dim=-2)

    if masked:
        zero_values = z_sum.W_C == 0
        z_sum.W_C[zero_values] = float("inf")
        z_sum.W_G[zero_values, :] = 0
        z_sum.W_Gp[zero_values, :] = 0

    z_soft = reciprocal(z_sum, **kwargs)

    return z_soft
