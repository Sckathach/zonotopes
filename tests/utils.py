from typing import Callable, Optional

import torch as t
from jaxtyping import Float
from torch import Tensor

from zonotope.zonotope import Zonotope


def empirical_soundness(
    a: Zonotope,
    result: Zonotope,
    concrete_fn: Callable,
    b: Optional[Zonotope] = None,
    n_points: int = 100,
    eps: float = 1e-5,
) -> None:
    lower, upper = result.concretize()
    for _ in range(n_points):
        if b is not None:
            sample = concrete_fn(a.sample_point(), b.sample_point())
        else:
            sample = concrete_fn(a.sample_point())
        assert t.all(lower - eps < sample)
        assert t.all(sample < upper + eps)


def check_bounds(
    z: Zonotope, expected_lower: Float[Tensor, "N"], expected_upper: Float[Tensor, "N"]
) -> None:
    lower, upper = z.concretize()
    assert t.allclose(lower, expected_lower.to(lower.device)), (
        f"Expected lower {expected_lower}, but got {lower}"
    )
    assert t.allclose(upper, expected_upper.to(upper.device)), (
        f"Expected upper {expected_upper}, but got {upper}"
    )
