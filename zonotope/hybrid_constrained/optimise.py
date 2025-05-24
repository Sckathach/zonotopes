import textwrap
from typing import Callable

import torch as t
from jaxtyping import Float
from torch import Tensor


def optimize_lambda(
    dims: t.Size | tuple,
    dual_fn: Callable[[Float[Tensor, "... J"]], Float[Tensor, "..."]],
    device: t.device,
    dtype: t.dtype,
    num_iterations: int = 1000,
    lr: float = 1e-3,
    verbose: bool = False,
) -> Float[Tensor, " "]:
    lmda = (
        t.randn(
            dims,
            device=device,
            dtype=dtype,
        )
        * 1e-4
    ).requires_grad_(True)

    optimizer = t.optim.Adam([lmda], lr=lr)

    best_lmda = lmda.clone().detach()
    best_value = float("-inf")

    for iteration in range(num_iterations):
        optimizer.zero_grad()

        current_bound = dual_fn(lmda)
        loss = -current_bound.sum()

        if t.isnan(loss).any():
            print(
                textwrap.dedent(f"""
                NaN detected at iteration {iteration}
                lmda stats: min={lmda.min().item()}, max={lmda.max().item()}
                concretize stats: {dual_fn(lmda)}
            """)
            )
            break

        loss.backward()

        # Gradient clipping to prevent exploding gradients
        t.nn.utils.clip_grad_norm_(lmda, max_norm=1.0)

        optimizer.step()

        # Tracking
        with t.no_grad():
            current_value = current_bound.sum().item()
            if current_value > best_value:
                best_value = current_value
                best_lmda = lmda.clone().detach()

        if iteration % 100 == 0 and verbose:
            print(f"Iteration {iteration}, Concretize Sum: {-loss.item()}")

    return best_lmda
