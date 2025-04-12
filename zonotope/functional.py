from typing import Tuple

import torch as t
from einops import einsum, rearrange, repeat
from jaxtyping import Float
from torch import Tensor
from torch.linalg import norm

from zonotope.zonotope import DUAL_INFINITY, Zonotope


def relu(zonotope: Zonotope) -> Zonotope:
    """
    Implements the ReLU abstract transformer for Zonotope domain (Section 4.3)
    """
    lower, upper = zonotope.concretize()

    result = zonotope.clone()

    case_neg = upper <= 0
    case_pos = lower >= 0
    case_cross = ~(case_neg | case_pos)

    # Case 1: Output is 0 when upper bound is negative
    result.W_C[case_neg] = 0
    result.W_Ei[case_neg] = 0
    result.W_Es[case_neg] = 0

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

        print(lambda_coeff, mu, beta_new)

        # Apply the transformation
        result.W_C[case_cross] = lambda_coeff * result.W_C[case_cross] + mu

        if result.Ei > 0:
            result.W_Ei[case_cross] = (
                lambda_coeff.unsqueeze(-1) * result.W_Ei[case_cross]
            )
        if result.Es > 0:
            result.W_Es[case_cross] = (
                lambda_coeff.unsqueeze(-1) * result.W_Es[case_cross]
            )

        # Add the new error term
        beta_new_reshaped = t.zeros(zonotope.N)
        beta_new_reshaped[case_cross] = beta_new
        result.W_Ei = t.cat([result.W_Ei, beta_new_reshaped.unsqueeze(-1)], dim=-1)

    return result


def tanh(zonotope: Zonotope) -> Zonotope:
    """
    Implements the tanh abstract transformer for the Zonotope domain (Section 4.4)
    """
    lower, upper = zonotope.concretize()
    result = zonotope.clone()

    tanh_l = t.tanh(lower)
    tanh_u = t.tanh(upper)

    # Calculate coefficients
    lambda_coeff = t.minimum(1 - tanh_l**2, 1 - tanh_u**2)
    mu = 0.5 * (tanh_u + tanh_l - lambda_coeff * (upper + lower))
    beta_new = 0.5 * (tanh_u - tanh_l - lambda_coeff * (upper - lower))

    # Apply the transformation
    result.W_C = lambda_coeff * result.W_C + mu

    if result.Ei > 0:
        result.W_Ei = lambda_coeff.unsqueeze(-1) * result.W_Ei
    if result.Es > 0:
        result.W_Es = lambda_coeff.unsqueeze(-1) * result.W_Es

    # Add new error term
    result.W_Ei = t.cat([result.W_Ei, beta_new.unsqueeze(-1)], dim=-1)

    return result


def exp(zonotope: Zonotope, eps_hat: float = 0.01, eps: float = 1e-6) -> Zonotope:
    """
    Implements the exponential abstract transformer for the Zonotope domain (Section 4.5)
    """
    lower, upper = zonotope.concretize()
    result = zonotope.clone()

    # Calculate exponential values
    exp_l = t.exp(lower)
    exp_u = t.exp(upper)

    # Calculate t_crit
    non_equal = (upper - lower) > eps
    t_crit = t.zeros_like(lower)

    # Where bounds differ significantly, use the formula
    t_crit[non_equal] = t.log(
        (exp_u[non_equal] - exp_l[non_equal]) / (upper[non_equal] - lower[non_equal])
    )

    # Where bounds are close, use the midpoint (approximation)
    t_crit[~non_equal] = (lower[~non_equal] + upper[~non_equal]) / 2

    # Calculate t_crit,2
    t_crit2 = lower + 1 - eps_hat

    # Calculate t_opt
    t_opt = t.minimum(t_crit, t_crit2)

    # Calculate coefficients
    lambda_coeff = t.exp(t_opt)
    mu = 0.5 * (t.exp(t_opt) - lambda_coeff * t_opt + exp_u - lambda_coeff * upper)
    beta_new = 0.5 * (
        lambda_coeff * t_opt - t.exp(t_opt) + exp_u - lambda_coeff * upper
    )

    # Apply the transformation
    result.W_C = lambda_coeff * result.W_C + mu
    if result.Ei > 0:
        result.W_Ei = lambda_coeff.unsqueeze(-1) * result.W_Ei
    if result.Es > 0:
        result.W_Es = lambda_coeff.unsqueeze(-1) * result.W_Es

    # Add new error term
    result.W_Ei = t.cat([result.W_Ei, beta_new.unsqueeze(-1)], dim=-1)

    return result


def reciprocal(zonotope: Zonotope, eps_hat: float = 0.01) -> Zonotope:
    """
    Implements the reciprocal abstract transformer for the Zonotope domain (Section 4.6)
    *(This is used for the softmax computation)*

    !Assumption: x > 0
    """
    lower, upper = zonotope.concretize()

    # Ensure all values are positive (for reciprocal to be well-defined)
    if t.any(lower <= 0):
        raise ValueError(
            "Reciprocal transformer requires all lower bounds to be positive"
        )

    result = zonotope.clone()

    # Calculate t_crit
    t_crit = t.sqrt(upper * lower)

    # Calculate t_crit,2
    t_crit2 = 0.5 * upper + eps_hat

    # Calculate t_opt
    t_opt = t.minimum(t_crit, t_crit2)

    # Calculate coefficients
    lambda_coeff = -1 / (t_opt**2)
    mu = 0.5 * (1 / t_opt - lambda_coeff * t_opt + 1 / lower - lambda_coeff * lower)
    beta_new = 0.5 * (
        lambda_coeff * t_opt - 1 / t_opt + 1 / lower - lambda_coeff * lower
    )

    # Apply the transformation
    result.W_C = lambda_coeff * result.W_C + mu
    if result.Ei > 0:
        result.W_Ei = lambda_coeff.unsqueeze(-1) * result.W_Ei
    if result.Es > 0:
        result.W_Es = lambda_coeff.unsqueeze(-1) * result.W_Es

    # Add new error term
    result.W_Ei = t.cat([result.W_Ei, beta_new.unsqueeze(-1)], dim=-1)

    return result


def dot_product(a: Zonotope, b: Zonotope) -> Zonotope:
    """
    Computes the dot product between two Zonotope vectors (Section 4.8).

    This implements both the fast and precise variants from the DeepT paper:
    - Fast variant: Uses dual norm trick for all types of terms
    - Precise variant: Uses more precise bounds for infinity terms
    """
    assert a.W_C.shape == b.W_C.shape, "Shapes must match for dot product"
    assert a.W_Ei.shape == b.W_Ei.shape, "Shapes must match for dot product"

    def _fast_bounds(
        v: Float[t.Tensor, "N E1"],
        w: Float[t.Tensor, "N E2"],
        norm_v: float,
        norm_w: float,
    ) -> Tuple[float, float]:
        bound = float(
            norm(
                einsum(norm(w, ord=norm_w, dim=0), v.abs(), "N, N E1 -> E1"),
                ord=norm_v,
                dim=0,
            ).item()
        )
        return -bound, bound

    def _precise_bounds(
        v: Float[t.Tensor, "N Ei"], w: Float[t.Tensor, "N Ei"]
    ) -> Tuple[float, float]:
        """
        Compute more precise bounds for infinity terms interactions.
        This implements the method from Equation (6) in the paper.
        """
        terms = einsum(v, w, "N Ei, N Ej -> Ei Ej")
        diag_terms = terms.diag()
        off_diag_terms = terms - t.eye(v.shape[1]) * terms.diag()

        x = float(diag_terms.sum().item())
        y = float(off_diag_terms.sum().item())

        upper_bound = max(x + y, x - y, -y, y)
        lower_bound = min(x + y, x - y, -y, y)

        return lower_bound, upper_bound

    result = a.clone()

    # 1. Handle center term
    result.W_C = einsum(a.W_C, b.W_C, "N, N -> N")

    # 2. Handle the linear terms
    if a.Ei > 0:
        result.W_Ei = einsum(a.W_C, b.W_Ei, "N, N Ei -> N Ei") + einsum(
            b.W_C, a.W_Ei, "N, N Ei -> N Ei"
        )

    if a.Es > 0:
        result.W_Es = einsum(a.W_C, b.W_Es, "N, N Es -> N Es") + einsum(
            b.W_C, a.W_Es, "N, N Es -> N Es"
        )

    # 3. Handle mixed terms
    l_phi_phi, l_phi_eps, l_eps_phi, l_eps_eps = 0.0, 0.0, 0.0, 0.0
    u_phi_phi, u_phi_eps, u_eps_phi, u_eps_eps = 0.0, 0.0, 0.0, 0.0

    if a.Es > 0 and b.Ei > 0:
        l_phi_eps, u_phi_eps = _fast_bounds(
            a.W_Es, b.W_Ei, norm_v=a.q, norm_w=DUAL_INFINITY
        )
    if a.Ei > 0 and b.Es > 0:
        l_eps_phi, u_eps_phi = _fast_bounds(
            a.W_Ei, b.W_Es, norm_v=DUAL_INFINITY, norm_w=b.q
        )
    if a.Es > 0 and b.Es > 0:
        l_phi_phi, u_phi_phi = _fast_bounds(a.W_Es, b.W_Es, norm_v=a.q, norm_w=b.q)

    if a.Ei > 0 and b.Ei > 0:
        l_eps_eps, u_eps_eps = _precise_bounds(a.W_Ei, b.W_Ei)

    lower = l_phi_phi + l_phi_eps + l_eps_phi + l_eps_eps
    upper = u_phi_phi + u_phi_eps + u_eps_phi + u_eps_eps

    # Create new noise symbol
    if upper > lower:
        beta_new = (upper - lower) / 2
        mu = (upper + lower) / 2

        result.W_C = result.W_C + mu

        result.W_Ei = t.cat(
            [result.W_Ei, (t.ones(a.N) * beta_new).unsqueeze(-1)], dim=-1
        )

    return result


def _compute_diff(w: Float[Tensor, "N ..."]) -> Float[Tensor, "N N ..."]:
    """
    Will be used in the fast softmax implementation. The diff are computed at once, removing the for i in range(z.N) loop.

    Returns diff (Nj Ni ...)
        vi - vj = diff[j, i]

    Works with shape 0 ..., will output 0 0 ...
    """
    w_expanded = repeat(w, "... -> N ...", N=w.shape[0])
    return w_expanded - rearrange(w_expanded, "Ni Nj ... -> Nj Ni ...")


def softmax(zonotope: Zonotope) -> Zonotope:
    """
    Implements the softmax abstract transformer for Zonotope domain (Section 5.2)

    Current implementation:
        - Line per line to avoid building NxNxEi tensors.
        - Adds two error terms, one for the exponential and one for the reciprocal.
        - New error terms are shared among variables (over approx).
        - The resulting sofmax variables do not sum to one! Consider using softmax_refinement.
    """
    result = Zonotope(
        center=t.zeros(zonotope.N),
        infinity_terms=t.zeros(zonotope.N, zonotope.Ei + 2),
        special_terms=t.zeros(zonotope.N, zonotope.Es),
        special_norm=zonotope.p,
    )
    for i in range(zonotope.N):
        diff_i = Zonotope(
            center=zonotope.W_C - zonotope.W_C[i],
            infinity_terms=zonotope.W_Ei - zonotope.W_Ei[i],
            special_terms=zonotope.W_Es - zonotope.W_Es[i],
            special_norm=zonotope.p,
        )

        exp_i = exp(diff_i)

        sum_i = Zonotope(
            center=exp_i.W_C.sum(dim=0, keepdim=True),
            infinity_terms=exp_i.W_Ei.sum(dim=0, keepdim=True),
            special_terms=exp_i.W_Es.sum(dim=0, keepdim=True),
            special_norm=zonotope.p,
        )

        reciprocal_i = reciprocal(sum_i)

        result.W_C[i] = reciprocal_i.W_C[0]
        result.W_Ei[i] = reciprocal_i.W_Ei[0]
        result.W_Es[i] = reciprocal_i.W_Es[0]

    return result


def softmax_refinement(z: Zonotope) -> Zonotope:
    """
    Implements the three-step softmax refinement process (Section 5.3)
    """
    if z.N <= 1:
        return z

    result = z.clone()

    # Step 1: Refine y1 by imposing the constraint y1 = 1 - (y2 + ... + yN)
    # y1
    c1 = result.W_C[0]
    b1 = result.W_Ei[0]
    a1 = result.W_Es[0]

    # 1 - (y2 + ... + yN)
    c2 = 1 - result.W_C[1:].sum(dim=0)
    b2 = -result.W_Ei[1:].sum(dim=0)
    a2 = -result.W_Es[1:].sum(dim=0)

    # Find indice eps_k
    eps_k = None
    for k in range(result.Ei):
        if b2[k] - b1[k] != 0:
            eps_k = k
            break

    # ? If no suitable epsilon found, isn't it ok?
    if eps_k is None:
        raise ValueError("eps_k is None")

    # Calculate new coefficients for the refined variable y1'
    # Choosing a value for beta'k
    beta_prime_k = find_optimal_beta(b1, b2, a1, a2, eps_k)
    beta_prime_k = 0.05

    # Calculate new coefficients
    cp = c2 + (c2 - c1) * (beta_prime_k - b2[eps_k]) / (b2[eps_k] - b1[eps_k])

    bp = b2.clone()
    mask = t.ones_like(b2).bool()
    mask[eps_k] = False
    bp[mask] = b2[mask] + (b2[mask] - b1[mask]) * (beta_prime_k - b2[eps_k]) / (
        b2[eps_k] - b1[eps_k]
    )
    bp[eps_k] = beta_prime_k

    ap = a2.clone()
    if z.Es > 0:
        ap = a2 + (a2 - a1) * (beta_prime_k - b2[eps_k]) / (b2[eps_k] - b1[eps_k])

    print(b1, bp)

    # Update the refined variable y1'
    result.W_C[0] = cp
    result.W_Ei[0] = bp
    result.W_Es[0] = ap

    # Step 2: Refine all other variables y2, ..., yN
    # Get the substitution for epsilon_k
    beta_k_diff = b1[eps_k] - b2[eps_k]

    const_eps_k_substitution = (c1 - c2) / beta_k_diff

    # Calculate coefficients for other epsilon terms
    infinity_eps_k_substitution = t.zeros_like(b1)
    for j in range(len(b1)):
        if j != eps_k:
            infinity_eps_k_substitution[j] = (b1[j] - b2[j]) / beta_k_diff

    # Calculate coefficients for special terms
    special_coeffs = t.zeros_like(a1)
    for j in range(len(a1)):
        special_coeffs[j] = (a1[j] - a2[j]) / beta_k_diff

    # Apply the substitution to all other variables
    for i in range(1, z.N):
        result.W_C[i] = result.W_C[i] + result.W_Ei[i, eps_k] * const_eps_k_substitution

        # Update infinity terms except eps_k
        for j in range(result.Ei):
            if j != eps_k:
                result.W_Ei[i, j] = (
                    result.W_Ei[i, j]
                    + result.W_Ei[i, eps_k] * infinity_eps_k_substitution[j]
                )

        # Zero out the eps_k term since it's been substituted
        result.W_Ei[i, eps_k] = 0

        # Update special terms
        for j in range(result.Es):
            result.W_Es[i, j] = (
                result.W_Es[i, j] + result.W_Ei[i, eps_k] * special_coeffs[j]
            )

    return result


def refine_softmax_bounds(result: Zonotope) -> Zonotope:
    # Step 3: Tighten the bounds of the noise symbols
    # Calculate the sum constraint: S = 1 - sum_yi' = 0
    sum_constraint = Zonotope(
        center=t.tensor([1.0 - result.W_C.sum()], device=result.W_C.device),
        infinity_terms=-result.W_Ei.sum(dim=0, keepdim=True),
        special_terms=-result.W_Es.sum(dim=0, keepdim=True),
        special_norm=result.p,
    )

    # Tighten bounds for each infinity noise symbol with non-zero coefficient
    refined_bounds = {}
    for m in range(result.Ei):
        beta_m_s = sum_constraint.W_Ei[0, m]

        if beta_m_s != 0:
            # Calculate bounds [a_m, b_m] based on equations in Section 5.3
            norm_alpha_s = t.linalg.norm(sum_constraint.W_Es[0], ord=sum_constraint.q)
            norm_beta_s = t.linalg.norm(sum_constraint.W_Ei[0], ord=1) - abs(beta_m_s)

            a_m = (sum_constraint.W_C[0] - norm_alpha_s - norm_beta_s) / abs(beta_m_s)
            b_m = (sum_constraint.W_C[0] + norm_alpha_s + norm_beta_s) / abs(beta_m_s)

            if beta_m_s < 0:
                a_m, b_m = -b_m, -a_m

            # Intersect with [-1, 1]
            new_a_m = max(-1.0, a_m)
            new_b_m = min(1.0, b_m)

            if new_a_m > -1.0 or new_b_m < 1.0:
                refined_bounds[m] = (new_a_m, new_b_m)

    # Apply the refined bounds by mapping each noise symbol with tightened bounds
    for m, (a_m, b_m) in refined_bounds.items():
        # Remap eps_m from [a_m, b_m] to [-1, 1]
        # eps_m = (a_m + b_m)/2 + (b_m - a_m)/2 * eps_new_m
        mid = (a_m + b_m) / 2
        scale = (b_m - a_m) / 2

        # Update all variables
        result.W_C = result.W_C + result.W_Ei[:, m] * mid
        result.W_Ei[:, m] = result.W_Ei[:, m] * scale

    return result


def find_optimal_beta(
    b1: Float[t.Tensor, "N Ei"],
    b2: Float[t.Tensor, "N Ei"],
    a1: Float[t.Tensor, "N Es"],
    a2: Float[t.Tensor, "N Es"],
    eps_k: int,
):
    """
    Find the value of beta'_k that minimizes the sum of absolute values of the coefficients.
    This implements the minimization algorithm from Section A.1 of the paper.
    """
    # Calculate all possible candidate solutions for beta'_k
    candidates = []

    beta_k_diff = b2[eps_k] - b1[eps_k]
    if beta_k_diff == 0:
        raise ValueError("Beta k diff can not be zero")

    # Add candidates from infinity terms
    for j in range(len(b1)):
        if j != eps_k:
            candidates.append(-((b2[j] - b1[j]) / beta_k_diff) + b2[eps_k])

    # Add candidates from special terms
    for j in range(len(a1)):
        candidates.append(-((a2[j] - a1[j]) / beta_k_diff) + b2[eps_k])

    # Calculate the evaluation function for each candidate
    def evaluate(beta_k):
        # score = abs(c2 + (c2 - c1) * (beta_k - ei2[eps_k]) / beta_k_diff)
        score = 0

        for j in range(len(b1)):
            if j != eps_k:
                score += abs(
                    b2[j] + (b2[j] - b1[j]) * (beta_k - b2[eps_k]) / beta_k_diff
                )

        for j in range(len(a1)):
            score += abs(a2[j] + (a2[j] - a1[j]) * (beta_k - b2[eps_k]) / beta_k_diff)

        return score

    # Find the candidate with the minimum score
    min_score = float("inf")
    beta_prime_k = 0.0
    candidates.append(t.Tensor([0.05]))

    for candidate in candidates:
        score = evaluate(candidate)
        print(candidate, score)
        if score < min_score:
            min_score = score
            beta_prime_k = float(candidate.item())

    # Return the optimal beta'_k
    return beta_prime_k
