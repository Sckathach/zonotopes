from typing import Tuple

import torch as t
from einops import einsum
from jaxtyping import Float
from torch import Tensor

from zonotope.classical.z import Zonotope


def relu(z: Zonotope) -> Zonotope:
    """
    Implements the ReLU abstract transformer for Zonotope domain (Section 4.3)
    """
    lower, upper = z.concretize()

    result = z.clone()

    case_neg = upper <= 0
    case_pos = lower >= 0
    case_cross = ~(case_neg | case_pos)

    # Case 1: Output is 0 when upper bound is negative
    result[case_neg] = 0
    # result.W_C[case_neg] = 0
    # result.W_Ei[case_neg] = 0
    # result.W_Es[case_neg] = 0

    # Case 2: No change needed for inputs with lower bound positive
    # Already handled by clone

    # Case 3: For crossing zero inputs, apply the linear approximation
    if t.any(case_cross):
        l_cross = lower[case_cross]
        u_cross = upper[case_cross]

        # Calculate coefficients for the linear approximation
        lambda_coeff = u_cross / (u_cross - l_cross)
        mu = 0.5 * t.maximum(-lambda_coeff * l_cross, (1 - lambda_coeff) * u_cross)
        beta_new = 0.5 * t.maximum(
            -lambda_coeff * l_cross, (1 - lambda_coeff) * u_cross
        )

        # Apply the transformation
        result.W_C[case_cross] = lambda_coeff * result.W_C[case_cross] + mu

        if result.I > 0:
            result.W_G[case_cross] = lambda_coeff.unsqueeze(-1) * result.W_G[case_cross]

        # Add the new error term
        beta_new_reshaped = t.zeros_like(z.W_C)
        beta_new_reshaped[case_cross] = beta_new
        result.W_G = t.cat([result.W_G, beta_new_reshaped.unsqueeze(-1)], dim=-1)

    return result


def tanh(z: Zonotope) -> Zonotope:
    """
    Implements the tanh abstract transformer for the Zonotope domain (Section 4.4)
    """
    lower, upper = z.concretize()
    result = z.clone()

    tanh_l = t.tanh(lower)
    tanh_u = t.tanh(upper)

    # Calculate coefficients
    lambda_coeff = t.minimum(1 - tanh_l**2, 1 - tanh_u**2)
    mu = 0.5 * (tanh_u + tanh_l - lambda_coeff * (upper + lower))
    beta_new = 0.5 * (tanh_u - tanh_l - lambda_coeff * (upper - lower))

    # Apply the transformation
    result.W_C = lambda_coeff * result.W_C + mu

    if result.I > 0:
        result.W_G = lambda_coeff.unsqueeze(-1) * result.W_G

    # Add new error term
    result.W_G = t.cat([result.W_G, beta_new.unsqueeze(-1)], dim=-1)

    return result


def exp(
    z: Zonotope, eps_hat: float = 0.01, eps: float = 1e-6, replace_inf: bool = False
) -> Zonotope:
    """
    Implements the exponential abstract transformer for the Zonotope domain (Section 4.5)
    TODO: add an option choose transformer: either > 0 or min area
    """
    lower, upper = z.concretize()
    result = z.clone()

    infinite_negative_values = float("-inf") == z.W_C

    # Calculate exponential values
    exp_l = t.exp(lower)
    exp_u = t.exp(upper)

    infinite_positive_values = (float("inf") == exp_l) | (float("inf") == exp_u)

    # Calculate t_crit
    non_equal = (upper - lower) > eps
    t_crit = t.zeros_like(lower)

    # Where bounds differ significantly, use the formula
    t_crit[non_equal] = t.log(
        (exp_u[non_equal] - exp_l[non_equal]) / (upper[non_equal] - lower[non_equal])
    )

    # Where bounds are close, use the midpoint (approximation)
    t_crit[~non_equal] = (lower[~non_equal] + upper[~non_equal]) / 2
    if replace_inf:
        t_crit[upper == lower] = float("inf")  # Replace NaNs by infinity
        t_crit[t_crit == float("-inf")] = (0.5 * lower + 0.5 * upper)[
            t_crit == float("-inf")
        ]  # Replace -Inf that arise from the log to avg(l, u)

    # Calculate t_crit,2
    # t_crit2 = lower + 1 - eps_hat

    # Calculate t_opt
    # t_opt = t.minimum(t_crit, t_crit2)
    t_opt = t_crit

    if replace_inf and (t_opt == float("-inf")).any():
        t_opt = t.min(
            # t.min(t_crit, t_crit2), upper
            t_crit,
            upper,
        )  # Idea: has to be below L + 1 (and <= U)

    # Calculate coefficients
    lambda_coeff = t.exp(t_opt)
    mu = 0.5 * (t.exp(t_opt) - lambda_coeff * t_opt + exp_u - lambda_coeff * upper)
    beta_new = 0.5 * (
        lambda_coeff * t_opt - t.exp(t_opt) + exp_u - lambda_coeff * upper
    )

    # Apply the transformation
    result.W_C = lambda_coeff * result.W_C + mu
    if result.I > 0:
        result.W_G = lambda_coeff.unsqueeze(-1) * result.W_G

    # Add new error term
    result.W_G = t.cat([result.W_G, beta_new.unsqueeze(-1)], dim=-1)

    result.W_C[infinite_negative_values] = 0
    result.W_G[infinite_negative_values, :] = 0

    result.W_C[infinite_positive_values] = float("inf")
    result.W_G[infinite_positive_values, :] = 0

    # values like 87 endup nan bcs exp(87) = 6e+37 -> out of bounds when float/half?
    # values like 84 endup -inf????
    nan_values = (result.W_C != result.W_C) | (float("-inf") == result.W_C)
    result.W_C[nan_values] = float("inf")
    result.W_G[nan_values, :] = 0

    return result


def reciprocal(
    zonotope: Zonotope,
    eps_hat: float = 1e-6,
    epsilon: float = 1e-6,
    ignore_negative_values: bool = False,
) -> Zonotope:
    """
    Implements the reciprocal abstract transformer for the Zonotope domain (Section 4.6)
    *(This is used for the softmax computation)*

    !Assumption: x > 0
    """
    lower, upper = zonotope.concretize()
    non_equal = (upper - lower) > epsilon

    # Ensure all values are positive (for reciprocal to be well-defined)
    if t.any(lower + epsilon < 0):
        if not ignore_negative_values:
            raise ValueError(
                "Reciprocal transformer requires all lower bounds to be positive"
            )

        print("[!] WARNING: enforcing positive values for the reciprocal transformer")
        lower[lower < 0] = epsilon
        upper[upper < 0] = epsilon

    result = zonotope.clone()

    # Calculate t_crit
    t_crit = t.sqrt(upper * lower)

    # Calculate t_crit,2
    # t_crit2 = 0.5 * upper + eps_hat

    # Calculate t_opt
    # t_opt = t.minimum(t_crit, t_crit2)
    t_opt = t_crit

    # Calculate coefficients
    lambda_coeff = -1 / (t_opt**2)
    mu = 0.5 * (1 / t_opt - lambda_coeff * t_opt + 1 / lower - lambda_coeff * lower)
    beta_new = 0.5 * (
        lambda_coeff * t_opt - 1 / t_opt + 1 / lower - lambda_coeff * lower
    )

    lambda_coeff[~non_equal] = 1
    mu[~non_equal] = 0
    beta_new[~non_equal] = 0

    # Apply the transformation
    result.W_C = lambda_coeff * result.W_C + mu
    if result.I > 0:
        result.W_G = lambda_coeff.unsqueeze(-1) * result.W_G

    # Add new error term
    result.W_G = t.cat([result.W_G, beta_new.unsqueeze(-1)], dim=-1)

    result.W_C[~non_equal] = 1 / zonotope.W_C[~non_equal]

    return result


def dot_product(a: Zonotope, b: Zonotope, pattern: str) -> Zonotope:
    """
    Computes the dot product between two Zonotope vectors (Section 4.8).

    This implements both the fast and precise variants from the DeepT paper:
    - Fast variant: Uses dual norm trick for all types of terms
    - Precise variant: Uses more precise bounds for infinity terms
    """
    assert a.I == b.I, "Shapes must match for dot product"

    dims_a, dims_bc = pattern.split(",")
    dims_b, dims_c = dims_bc.split("->")

    def _precise_bounds(
        a_infinity_terms: Float[t.Tensor, "dims_a Ei"],
        b_infinity_terms: Float[t.Tensor, "dims_b Ei"],
    ) -> Tuple[Float[Tensor, "dims_c"], Float[Tensor, "dims_c"]]:
        """
        Compute more precise bounds for infinity terms interactions.
        This implements the method from Equation (6) in the paper.
        """
        terms = einsum(
            a_infinity_terms,
            b_infinity_terms,
            f"{dims_a} Ei, {dims_b} Ej -> {dims_c} Ei Ej",
        )
        x = terms.diagonal(dim1=-1, dim2=-2).sum(dim=-1)
        y = terms.sum(dim=[-1, -2]) - x

        upper_bound = t.maximum(t.maximum(t.maximum(x + y, x - y), -y), y)
        lower_bound = t.minimum(t.minimum(t.minimum(x + y, x - y), -y), y)

        return lower_bound, upper_bound

    result = a.clone()

    # 1. Handle center term
    result.W_C = einsum(a.W_C, b.W_C, pattern)

    # 2. Handle the linear terms
    if a.I > 0:
        result.W_G = einsum(
            a.W_C, b.W_G, f"{dims_a}, {dims_b} Ei -> {dims_c} Ei"
        ) + einsum(b.W_C, a.W_G, f"{dims_b}, {dims_a} Ei -> {dims_c} Ei")

    # 3. Handle mixed terms
    lower = t.zeros_like(result.W_C)
    upper = t.zeros_like(result.W_C)

    if a.I > 0 and b.I > 0:
        l_eps_eps, u_eps_eps = _precise_bounds(a.W_G, b.W_G)
        lower += l_eps_eps
        upper += u_eps_eps

    # Create new noise symbol
    if not t.all(lower - upper == 0):
        beta_new = (upper - lower) / 2
        mu = (upper + lower) / 2

        result.W_C += mu

        result.W_G = (
            t.cat(
                [result.W_G, beta_new.unsqueeze(-1)],
                dim=-1,
            )
            if result.I > 0
            else beta_new.unsqueeze(-1)
        )

    return result.clone()


def replace_invalid_values(
    z: Zonotope, replace_value: float, reset_error_terms: bool = True
) -> Zonotope:
    result = z.clone()

    mask = (z.W_C != z.W_C) | (float("inf") == z.W_C) | (float("-inf") == z.W_C)
    result.W_C[mask] = replace_value

    if reset_error_terms:
        result.W_G[mask, :] = 0

    return result


def softmax(
    z: Zonotope, masked: bool = True, ignore_negative_values: bool = True
) -> Zonotope:
    a = z.repeat("... -> ... N", N=z.W_C.shape[-1])
    b = a.rearrange("... Ni Nj -> ... Nj Ni")
    z_diff = a - b

    if masked:
        z_diff = replace_invalid_values(z_diff, replace_value=float("-inf"))

    z_exp = exp(z_diff)
    z_sum = z_exp.sum(dim=-2)

    if masked:
        zero_values = z_sum.W_C == 0
        z_sum.W_C[zero_values] = float("inf")
        z_sum.W_G[zero_values, :] = 0

    z_soft = reciprocal(z_sum, ignore_negative_values=ignore_negative_values)

    return z_soft
