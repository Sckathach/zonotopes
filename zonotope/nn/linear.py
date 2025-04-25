from jaxtyping import Float
from torch import Tensor

from zonotope.functional import relu
from zonotope.nn.concrete.mlp import MLP
from zonotope.zonotope import Zonotope


def linear(z: Zonotope, weight: Float[Tensor, "out in"], bias: Tensor) -> Zonotope:
    return z.einsum(weight, "... in, out in -> ... out") + bias


def mlp(z: Zonotope, concrete_model: MLP):
    layers = concrete_model.layers
    for layer in layers[:-1]:
        z = linear(z, layer.weight, layer.bias)  # type: ignore
        z = relu(z)
    z = linear(z, layers[-1].weight, layers[-1].bias)  # type: ignore

    return z
