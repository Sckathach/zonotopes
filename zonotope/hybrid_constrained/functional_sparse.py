from functools import partial
from typing import Callable

import torch as t
from jaxtyping import Float
from torch import Tensor

from zonotope.classical.functional import exp as classical_exp
from zonotope.classical.functional import reciprocal as classical_reciprocal
from zonotope.classical.functional import tanh as classical_tanh
from zonotope.classical.z import Zonotope
from zonotope.hybrid_constrained.hcz_sparse import HCZ


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
    if t.allclose(t.as_tensor(lower), t.as_tensor(upper)):
        z = Zonotope.from_values(center=t.as_tensor([lower]))
    else:
        z = Zonotope.from_bounds(lower=t.as_tensor([lower]), upper=t.as_tensor([upper]))
    r = classical_zonotope_abstract_transformer(z)

    if z.Ei > 0 and r.Ei > 0:
        return HCZ.from_values(
            [z.W_C[0].item(), r.W_C[0].item()],
            t.tensor([[z.W_Ei[0, 0].item(), 0], r.W_Ei.tolist()[0]]).T,
        )

    return HCZ.from_values(
        [z.W_C[0].item(), r.W_C[0].item()],
    )


def abstract_relu(lower: Tensor | float, upper: Tensor | float) -> HCZ:
    if lower >= 0:
        return HCZ.from_values(
            [(upper + lower) / 2, (upper + lower) / 2],
            t.tensor([[(upper - lower) / 2], [(upper - lower) / 2]]).T,
        )
    if upper <= 0:
        return HCZ.from_values(
            [(upper + lower) / 2, 0], t.tensor([[(upper - lower) / 2], [0]]).T
        )

    h1 = HCZ.from_values([lower / 2, 0], t.tensor([[lower / 2], [0]]).T)
    h2 = HCZ.from_values([upper / 2, upper / 2], t.tensor([[upper / 2], [upper / 2]]).T)
    return h1.union(h2, check_emptiness=False)


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
    return h1.union(h2, check_emptiness=False)


def apply_abstract_transformer(
    z_in: HCZ,
    abs_transformer: Callable[[Tensor | float, Tensor | float], HCZ],
    **kwargs,
) -> HCZ:
    lower, upper = z_in.concretize(**kwargs)
    z_abs = abs_transformer(lower[0], upper[0])

    for i in range(1, z_in.N):
        z_abs = z_abs.cartesian_product(
            abs_transformer(lower[i], upper[i]), check_emptiness=False
        )

    z_abs.load_config_from_(z_in)
    perm = get_permutation_matrix(z_abs)
    r_in = t.cat([z_in.eye(z_in.N), z_in.zeros(z_in.N, z_in.N)], dim=-1)
    perm_z_abs = z_abs.mm(perm, check_emptiness=False)
    intermediate_result = perm_z_abs.intersect(
        z_in,
        r_in.T,
        check_emptiness_before=False,
        check_emptiness_after=False,
        **kwargs,
    )
    r_out = t.cat([z_in.zeros(z_in.N, z_in.N), z_in.eye(z_in.N)], dim=-1)
    return intermediate_result.mm(r_out.T, check_emptiness=False)


def linear(z: HCZ, weight: Float[Tensor, "out in"], bias: Tensor) -> HCZ:
    return z.mm(weight.T) + bias


def dot_product(a: HCZ, b: HCZ, **kwargs) -> HCZ:
    h1, h2 = a.clone(), b.clone()

    l1, u1 = h1.concretize(**kwargs)
    l2, u2 = h2.concretize(**kwargs)

    lhat = (l1 - h1.W_C) @ (l2 - h2.W_C)
    uhat = (u1 - h1.W_C) @ (u2 - h2.W_C)

    return h1.clone(
        W_c=h1.W_C.T @ h2.W_C,
        W_G=h1.cat(
            [t.sparse.mm(h1.W_G, h2.W_C.unsqueeze(-1).to_sparse_coo())],
            [t.sparse.mm(h2.W_G, h1.W_C.unsqueeze(-1).to_sparse_coo())],
            [lhat.unsqueeze(0).to_sparse_coo()],
            [(1, h1.N)],
            [uhat.unsqueeze(0).to_sparse_coo()],
            [(1, h1.N)],
        ),
        W_Gp=h1.cat(
            [t.sparse.mm(h1.W_Gp, h2.W_C.unsqueeze(-1).to_sparse_coo())],
            [t.sparse.mm(h2.W_Gp, h1.W_C.unsqueeze(-1).to_sparse_coo())],
        ),
        W_A=h1.cat(
            [h1.W_A, (h1.I, h2.J + 2)],
            [(h2.I, h1.J), h2.W_A, (h2.I, 2)],
            [(4, h1.J + h2.J), h1.as_sparse_tensor([[1, 0], [1, 0], [0, 1], [0, 1]])],
        ),
        W_Ap=h1.cat([h1.W_Ap, (h1.Ip, h2.J + 2)], [(h2.Ip, h1.J), h2.W_A, (h2.Ip, 2)]),
        W_b=h1.cat([h1.W_B, h2.W_B, h1.as_tensor([-1, 1])]),
    )
