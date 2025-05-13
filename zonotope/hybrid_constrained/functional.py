import torch as t

from zonotope.hybrid_constrained.hcz import HCZ


def relu(z: HCZ, **kwargs) -> HCZ:
    r = z.clone()
    _, upper = z.concretize(**kwargs)
    mask_zeros = upper <= 0
    r[mask_zeros] = 0
    y = HCZ.from_bounds(t.zeros_like(z.W_C), t.max(upper, t.zeros_like(upper)))
    return r.intersect(y)


def reciprocal(z: HCZ, epsilon: float = 1e-6, eps_hat: float = 1e-6, **kwargs) -> HCZ:
    result = z.clone()
    lower, upper = z.concretize(**kwargs)

    non_equal = (upper - lower).abs() > epsilon

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

    lambda_coeff[~non_equal] = 1
    mu[~non_equal] = 0
    beta_new[~non_equal] = 0

    # Apply the transformation
    result.W_C = lambda_coeff * result.W_C + mu
    if result.Ng > 0:
        result.W_Gc = lambda_coeff.unsqueeze(-1) * result.W_Gc
    if result.Nb > 0:
        result.W_Gb = lambda_coeff.unsqueeze(-1) * result.W_Gb

    # Add new error term
    result.W_Gc = t.cat([result.W_Gc, beta_new.unsqueeze(-1)], dim=-1)

    result.W_C[~non_equal] = 1 / z.W_C[~non_equal]
    result.W_Ac = t.cat([result.W_Ac, result.zeros(result.Nc, 1)], dim=-1)

    return result
