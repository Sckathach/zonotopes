import math
from typing import Any, Dict, List, Optional

import torch as t


def get_dim_for_error_terms(dim: int | List[int]) -> int | List[int]:
    if isinstance(dim, int):
        return dim - 1 if dim < 0 else dim

    return [d - 1 if d < 0 else d for d in dim]


def dual_norm(p: float) -> float:
    if math.isinf(p):
        return 1

    return float("inf") if p == 1 else p / (p - 1)


def parse_einops_pattern(
    pattern: str, shape_a: Optional[t.Size] = None, shape_b: Optional[t.Size] = None
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}

    if "->" not in pattern:
        raise ValueError(f"Invalid einsum pattern: {pattern}")

    if "," not in pattern:
        dims_a, dims_b = pattern.split("->")
        return {
            "pattern_in": dims_a,
            "pattern_out": dims_b,
        }

    dims_a, dims_bc = pattern.split(",")
    dims_b, dims_c = dims_bc.split("->")

    result = result | {
        "pattern_left_in": dims_a,
        "pattern_right_in": dims_b,
        "pattern_out": dims_c,
    }

    if shape_a is None or shape_b is None:
        return result

    dims_a_splited = dims_a.strip().split(" ")
    dims_b_splited = dims_b.strip().split(" ")
    dims_c_splited = dims_c.strip().split(" ")

    d_a = {k: v for k, v in zip(dims_a_splited, shape_a, strict=True)}
    d_b = {k: v for k, v in zip(dims_b_splited, shape_b, strict=True)}
    d_c = d_a | d_b
    removed_dim = list(d_a.keys() & d_b.keys())[0]
    d_c.pop(removed_dim)
    out_shape = t.Size([d_c[k] for k in dims_c_splited])
    removed_dim_pos = dims_a_splited.index(removed_dim)

    return result | {
        "out_shape": out_shape,
        "dict_left_in": d_a,
        "dict_right_in": d_b,
        "dict_out": d_c,
        "removed_dim": removed_dim_pos,
    }


def get_einops_pattern_for_error_terms(pattern: str) -> str:
    patterns = parse_einops_pattern(pattern)

    if len(patterns) == 2:
        return f"{patterns['pattern_in']} E -> {patterns['pattern_out']} E"

    return f"{patterns['pattern_left_in']} E, {patterns['pattern_right_in']} -> {patterns['pattern_out']} E"
