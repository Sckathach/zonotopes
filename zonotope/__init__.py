import gc

import torch as t

ORANGE = "#FFB84C"
VIOLET = "#A459D1"
PINK = "#F266AB"
TURQUOISE = "#2CD3E1"
DEVICE: str = "cuda" if t.cuda.is_available() else "cpu"

INFINITY = float("inf")
EPSILON = 1e-12


def dual_norm(p: float) -> float:
    if p == 1:
        return INFINITY
    elif p == 2:
        return 2.0
    elif p > 10:  # represents the infinity norm
        return 1.0
    else:
        raise NotImplementedError(
            "dual_norm: Dual norm only supported for 1-norm (p = 1), 2-norm (p = 2) or inf-norm (p > 10)"
        )


DUAL_INFINITY = dual_norm(INFINITY)


def cleanup_memory() -> None:
    gc.collect()
    t.cuda.empty_cache()
