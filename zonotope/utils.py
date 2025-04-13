from typing import Any, Callable, Optional

import torch as t
from einops import einsum
from jaxtyping import Float
from torch import Tensor

from zonotope.zonotope import Zonotope


def create_zonotope(
    center_values: Any,
    infinity_terms: Any = None,
    special_terms: Any = None,
    p: int = 2,
) -> Zonotope:
    center = t.tensor(center_values, dtype=t.float16)

    if infinity_terms is not None:
        infinity_terms = t.tensor(infinity_terms, dtype=t.float16)

    if special_terms is not None:
        special_terms = t.tensor(special_terms, dtype=t.float16)

    return Zonotope(
        center=center,
        infinity_terms=infinity_terms,
        special_terms=special_terms,
        special_norm=p,
    )


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


def sample_infinite(z: Zonotope) -> Float[Tensor, "S N"]:
    points = [[-1], [1]]
    for _ in range(0, z.Ei - 1):
        new_points = []
        new_points.extend([x + [-1] for x in points])
        new_points.extend([x + [1] for x in points])
        points = new_points

    eps = t.tensor(points, dtype=z.dtype).to(z.device)
    samples = einsum(z.W_Ei, eps, "N Ei, S Ei -> S N")
    return z.W_C.unsqueeze(0).repeat(samples.shape[0], 1) + samples
