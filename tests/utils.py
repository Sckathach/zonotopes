from typing import Any, Callable, Optional, cast

import einx
import torch as t
from torch import Tensor

from zonotope.classical.z import Zonotope


def empirical_soundness(
    a: Zonotope,
    result: Zonotope,
    concrete_fn: Callable,
    b: Optional[Zonotope] = None,
    n_points: int = 100,
    eps: float = 1e-5,
) -> None:
    lower, upper = result.concretize()
    if b is not None:
        sample = concrete_fn(
            a.sample_point(n_points),
            b.sample_point(n_points),
        )
    else:
        sample = concrete_fn(a.sample_point())

    lower = cast(
        Tensor, einx.rearrange("... -> n_points ...", lower, n_points=n_points)
    )
    upper = cast(
        Tensor, einx.rearrange("... -> n_points ...", upper, n_points=n_points)
    )

    assert t.all(lower - eps < sample)
    assert t.all(sample < upper + eps)


def check_bounds(z: Zonotope, expected_lower: Any, expected_upper: Any) -> None:
    lower, upper = z.concretize()
    assert t.allclose(lower, z.as_tensor(expected_lower)), (
        f"Expected lower {expected_lower}, but got {lower}"
    )
    assert t.allclose(upper, z.as_tensor(expected_upper)), (
        f"Expected upper {expected_upper}, but got {upper}"
    )
