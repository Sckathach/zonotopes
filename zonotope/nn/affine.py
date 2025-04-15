import torch as t

from zonotope.zonotope import Zonotope


def affine_transformer(z: Zonotope, weight: t.Tensor, bias: t.Tensor) -> Zonotope:
    return Zonotope(
        center=t.matmul(weight, z.W_C) + bias,
        infinity_terms=t.matmul(weight, z.W_Ei),
        special_terms=t.matmul(weight, z.W_Es),
        special_norm=z.p,
    )
