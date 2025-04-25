import math
from typing import List


def get_einops_pattern_for_error_terms(pattern: str) -> str:
    if "->" not in pattern:
        raise ValueError(f"Invalid einsum pattern: {pattern}")

    if "," not in pattern:
        dims_a, dims_b = pattern.split("->")
        return f"{dims_a} E -> {dims_b} E"

    dims_a, dims_bc = pattern.split(",")
    dims_b, dims_c = dims_bc.split("->")
    return f"{dims_a} E, {dims_b} -> {dims_c} E"


def get_dim_for_error_terms(dim: int | List[int]) -> int | List[int]:
    if isinstance(dim, int):
        return dim - 1 if dim < 0 else dim

    return [d - 1 if d < 0 else d for d in dim]


def dual_norm(p: float) -> float:
    if math.isinf(p):
        return 1

    return float("inf") if p == 1 else p / (p - 1)
