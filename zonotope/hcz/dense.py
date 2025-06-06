from functools import partial
from typing import Any, Optional, Self, Tuple

import torch as t
from einops import einsum, rearrange, repeat
from jaxtyping import Float
from torch import Tensor
from torch.linalg import norm

from zonotope import DEFAULT_DEVICE, DEFAULT_DTYPE
from zonotope.hcz.base import HCZBase, HCZConfig
from zonotope.hcz.optimise import optimize_lambda
from zonotope.utils import get_dim_for_error_terms, get_einops_pattern_for_error_terms


class HCZDense(HCZBase):
    def __init__(
        self,
        W_C: Float[Tensor, "..."],
        W_G: Optional[Float[Tensor, "I ..."]] = None,
        W_Gp: Optional[Float[Tensor, "Ip ..."]] = None,
        W_A: Optional[Float[Tensor, "I J"]] = None,
        W_Ap: Optional[Float[Tensor, "Ip J"]] = None,
        W_B: Optional[Float[Tensor, "J"]] = None,
        config: Optional[HCZConfig] = None,
        clone: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(W_C, W_G, W_Gp, W_A, W_Ap, W_B, config, clone, **kwargs)

    @classmethod
    def from_values(
        cls,
        W_C: Any,
        W_G: Any = None,
        W_Gp: Any = None,
        W_A: Any = None,
        W_Ap: Any = None,
        W_B: Any = None,
        dtype: t.dtype = DEFAULT_DTYPE,
        device: t.device = DEFAULT_DEVICE,
        config: Optional[HCZConfig] = None,
        **kwargs,
    ) -> Self:
        as_tensor = partial(t.as_tensor, dtype=dtype, device=device)

        return cls(
            W_c=as_tensor(W_C),
            W_G=as_tensor(W_G) if W_G is not None else None,
            W_Gp=as_tensor(W_Gp) if W_Gp is not None else None,
            W_A=as_tensor(W_A) if W_A is not None else None,
            W_Ap=as_tensor(W_Ap) if W_Ap is not None else None,
            W_b=as_tensor(W_B) if W_B is not None else None,
            config=config,
            **kwargs,
        )

    @classmethod
    def from_bounds(
        cls,
        lower: Any,
        upper: Any,
        dtype: t.dtype = DEFAULT_DTYPE,
        device: t.device = DEFAULT_DEVICE,
    ) -> "HCZDense":
        lower_ = t.as_tensor(lower, dtype=dtype, device=device)
        upper_ = t.as_tensor(upper, dtype=dtype, device=device)

        center = (lower_ + upper_) / 2
        radius = (upper_ - lower_) / 2

        radius_flat = radius.reshape(-1)
        radius_flat_nonzero = radius_flat[radius_flat != 0]
        N: int = radius_flat_nonzero.shape[0]
        radius_eye_non_zero = t.eye(N, dtype=dtype, device=device) * radius_flat_nonzero
        radius_expanded = t.zeros(radius_flat.shape[0], N, dtype=dtype, device=device)
        radius_expanded[radius_flat != 0] = radius_eye_non_zero
        radius_reshaped = radius_expanded.reshape(*center.shape, N)
        return cls.from_values(W_C=center, W_G=radius_reshaped)

    def concretize(self, **kwargs) -> Tuple[Float[Tensor, "..."], Float[Tensor, "..."]]:
        """
        Returns: lower, upper
        """
        if self.J == 0:
            return self.dual_lower(self.zeros(*self.shape, self.J)), -self.dual_upper(
                self.zeros(*self.shape, self.J)
            )

        kwargs = {"lr": self.config.lr, "n_steps": self.config.n_steps} | kwargs
        lambda_lower = optimize_lambda(
            (*self.shape, self.J),
            self.dual_lower,
            device=self.device,
            dtype=self.dtype,
            **kwargs,
        )
        lambda_upper = optimize_lambda(
            (*self.shape, self.J),
            self.dual_upper,
            device=self.device,
            dtype=self.dtype,
            **kwargs,
        )

        return self.dual_lower(lambda_lower), -self.dual_upper(lambda_upper)

    def dual_lower(self, lmda: Float[Tensor, "... j"]) -> Float[Tensor, "..."]:
        return (
            self.W_C
            + einsum(lmda, self.W_B, "... j, j -> ...")
            - norm(
                self.W_G - einsum(lmda, self.W_A, "... j, j i -> ... i"),
                ord=1,
                dim=-1,
            )
            - norm(
                self.W_Gp - einsum(lmda, self.W_Ap, "... j, j ip -> ... ip"),
                ord=1,
                dim=-1,
            )
        )

    def dual_upper(self, lmda: Float[Tensor, "... j"]) -> Float[Tensor, "..."]:
        return (
            -self.W_C
            + einsum(lmda, self.W_B, "... j, j -> ...")
            - norm(
                -self.W_G - einsum(lmda, self.W_A, "... j, j i -> ... i"),
                ord=1,
                dim=-1,
            )
            - norm(
                -self.W_Gp - einsum(lmda, self.W_Ap, "... j, j ip -> ... ip"),
                ord=1,
                dim=-1,
            )
        )

    def add(self, other: Self | float | int | Tensor) -> Self:  # type: ignore
        if isinstance(other, HCZDense):
            return self.clone(
                W_c=self.W_C + other.W_C,
                W_G=self.cat([self.W_G, other.W_G]),
                W_Gp=self.cat([self.W_Gp, other.W_Gp]),
                W_A=self.cat(
                    [self.W_A, (self.J, other.I)],
                    [(other.J, self.I), other.W_A],
                ),
                W_Ap=self.cat(
                    [self.W_Ap, (self.J, other.Ip)], [(other.J, self.Ip), other.W_Ap]
                ),
                W_b=self.cat([self.W_B, other.W_B]),
            )

        return self.clone(W_c=self.W_C + other)

    def mul(self, other: float | int | Tensor) -> Self:
        if isinstance(other, Tensor):
            return self.clone(
                W_c=self.W_C * other,
                W_G=self.W_G * other.unsqueeze(-1),
                W_Gp=self.W_Gp * other.unsqueeze(-1),
            )
        else:
            return self.clone(
                W_c=self.W_C * other,
                W_G=self.W_G * other,
                W_Gp=self.W_Gp * other,
            )

    def intersect(
        self, other: Self, r: Optional[Float[Tensor, "N2 N1"]] = None, **kwargs
    ) -> Self:
        if self.N == 0 or other.N == 0:
            return self.empty_from_self()

        w_c_flat = self.W_C.clone().reshape(-1)
        w_g_flat = self.W_G.clone().reshape(self.N, self.I)
        w_gp_flat = self.W_Gp.clone().reshape(self.N, self.Ip)
        o_c_flat = other.W_C.clone().reshape(-1)
        o_g_flat = other.W_G.clone().reshape(other.N, other.I)
        o_gp_flat = other.W_Gp.clone().reshape(other.N, other.Ip)

        if r is not None:
            w_c_flat = einsum(r, w_c_flat, "n2 n1, n1 -> n2")
            w_g_flat = einsum(r, w_g_flat, "n2 n1, n1 i -> n2 i")
            w_gp_flat = einsum(r, w_gp_flat, "n2 n1, n1 ip -> n2 ip")

        new_b = [[self.W_B], [other.W_B]]
        if self.I != 0 or self.Ip != 0 or other.I != 0 or other.Ip != 0:
            new_b.append([w_c_flat - o_c_flat])

        result = self.clone(
            W_G=self.cat([self.W_G, (*self.shape, other.I)]),
            W_Gp=self.cat([self.W_Gp, (*self.shape, other.Ip)]),
            W_A=self.cat(
                [self.W_A, (self.J, other.I)],
                [(other.J, self.I), other.W_A],
                [w_g_flat, -o_g_flat],
            ),
            W_Ap=self.cat(
                [self.W_Ap, (self.J, other.Ip)],
                [(other.J, self.Ip), other.W_Ap],
                [w_gp_flat, -o_gp_flat],
            ),
            W_b=self.cat(*new_b),  # type: ignore
        )
        if result.is_empty(**kwargs):
            return self.empty_from_self()

        return result

    def is_empty(self, **kwargs) -> bool:
        lower, upper = self.concretize(**kwargs)
        return bool(t.any(lower > upper).item())

    def cartesian_product(self, other: Self) -> Self:
        return self.clone(
            W_c=self.cat([self.W_C], [other.W_C]),
            W_G=self.cat(
                [self.W_G, (*self.shape, other.I)], [(*other.shape, self.I), other.W_G]
            ),
            W_Gp=self.cat(
                [self.W_Gp, (*self.shape, other.Ip)],
                [(*other.shape, self.Ip), other.W_Gp],
            ),
            W_A=self.cat([self.W_A, (self.J, other.I)], [(other.J, self.I), other.W_A]),
            W_Ap=self.cat(
                [self.W_Ap, (self.J, other.Ip)], [(other.J, self.Ip), other.W_Ap]
            ),
            W_b=self.cat([self.W_B], [other.W_B]),
        )

    def union(self, other: Self) -> Self:
        if self.N == 0:
            return other

        if other.N == 0:
            return self

        I1, I2, Ip1, Ip2, J1, J2 = self.I, other.I, self.Ip, other.Ip, self.J, other.J
        Inew = 2 * I1 + 2 * Ip1 + 2 * I2 + 2 * Ip2

        return self.clone(
            W_c=1 / 2 * (self.W_C + other.W_C + self.W_Gp.sum(-1) + other.W_Gp.sum(-1)),
            W_Gp=self.cat(
                [
                    self.W_Gp,
                    other.W_Gp,
                    1
                    / 2
                    * (
                        self.W_C - other.W_C + self.W_Gp.sum(-1) - other.W_Gp.sum(-1)
                    ).unsqueeze(-1),
                ]
            ),
            W_G=self.cat([self.W_G, other.W_G, self.zeros(*self.shape, Inew)]),
            W_A=self.cat(
                [self.W_A, (J1, I2 + Inew)],
                [(J2, I1), other.W_A, (J2, Inew)],
                [
                    self.cat(
                        [
                            self.eye(I1),
                            -self.eye(I1),
                            (Inew - 2 * I1, I1),
                        ],
                        [
                            (2 * I1, I2),
                            self.eye(I2),
                            -self.eye(I2),
                            (2 * Ip1 + 2 * Ip2, I2),
                        ],
                        [self.eye(Inew)],
                        row_dims=-1,
                        column_dims=0,
                    )
                ],
            ),
            W_Ap=self.cat(
                [
                    self.W_Ap,
                    (J1, Ip2),
                    -1 / 2 * (self.W_B + self.W_Ap.sum(-1)).unsqueeze(-1),
                ],
                [
                    (J2, Ip1),
                    other.W_Ap,
                    1 / 2 * (other.W_B + other.W_Ap.sum(-1)).unsqueeze(-1),
                ],
                [
                    1
                    / 2
                    * self.cat(
                        [
                            (2 * I1 + 2 * I2, Ip1),
                            self.eye(Ip1),
                            -self.eye(Ip1),
                            (2 * Ip2, Ip1),
                        ],
                        [
                            (2 * I1 + 2 * I2 + 2 * Ip1, Ip2),
                            self.eye(Ip2),
                            -self.eye(Ip2),
                        ],
                        [
                            self.ones(2 * I1, 1),
                            -self.ones(2 * I2, 1),
                            self.ones(2 * Ip1, 1),
                            -self.ones(2 * Ip2, 1),
                        ],
                        row_dims=-1,
                        column_dims=0,
                    )
                ],
            ),
            W_b=self.cat(
                [1 / 2 * (self.W_B - self.W_Ap.sum(-1))],
                [1 / 2 * (other.W_B - other.W_Ap.sum(-1))],
                [
                    1 / 2 * self.ones(2 * I1 + 2 * I2),
                    (Ip1,),
                    self.ones(Ip1),
                    (Ip2,),
                    self.ones(Ip2),
                ],
            ),
        )

    def rearrange(self, pattern: str, **kwargs) -> Self:
        """Einops rearrange"""
        error_pattern = get_einops_pattern_for_error_terms(pattern)
        return self.clone(
            W_c=rearrange(self.W_C, pattern, **kwargs),
            W_G=rearrange(self.W_G, error_pattern, **kwargs),
            W_Gp=rearrange(self.W_Gp, error_pattern, **kwargs),
        )

    def repeat(self, pattern: str, **kwargs) -> Self:
        """Einops repeat"""
        error_pattern = get_einops_pattern_for_error_terms(pattern)
        return self.clone(
            W_c=repeat(self.W_C, pattern, **kwargs),
            W_G=repeat(self.W_G, error_pattern, **kwargs),
            W_Gp=repeat(self.W_Gp, error_pattern, **kwargs),
        )

    def einsum(self, other: Tensor, pattern: str, **kwargs) -> Self:
        """Einops einsum"""
        error_pattern = get_einops_pattern_for_error_terms(pattern)
        return self.clone(
            W_c=einsum(self.W_C, other, pattern, **kwargs),
            W_G=einsum(self.W_G, other, error_pattern, **kwargs),
            W_Gp=einsum(self.W_Gp, other, error_pattern, **kwargs),
        )

    def sum(self, dim: int, **kwargs) -> Self:
        error_dim = get_dim_for_error_terms(dim)
        return self.clone(
            W_c=self.W_C.sum(dim=dim, **kwargs),
            W_G=self.W_G.sum(dim=error_dim, **kwargs) if self.I > 0 else None,
            W_Gp=self.W_Gp.sum(dim=error_dim, **kwargs) if self.Ip > 0 else None,
        )

    def mean(self, dim: int, **kwargs) -> Self:
        error_dim = get_dim_for_error_terms(dim)
        return self.clone(
            W_c=self.W_C.mean(dim=dim, **kwargs),
            W_G=self.W_G.mean(dim=error_dim, **kwargs) if self.I > 0 else None,
            W_Gp=self.W_Gp.mean(dim=error_dim, **kwargs) if self.Ip > 0 else None,
        )

    def reshape(self, *shape) -> Self:
        return self.clone(
            W_c=self.W_C.reshape(*shape),
            W_G=self.W_G.reshape(*shape, self.I) if self.I > 0 else None,
            W_Gp=self.W_Gp.reshape(*shape, self.Ip) if self.Ip > 0 else None,
        )

    def contiguous(self) -> None:
        """Torch contiguous"""
        self.W_C.contiguous()
        self.W_G.contiguous()
        self.W_Gp.contiguous()
        self.W_A.contiguous()
        self.W_Ap.contiguous()
        self.W_B.contiguous()

    def __getitem__(self, key) -> Self:
        if isinstance(key, tuple):
            error_key = (*key, slice(None, None, None))
        else:
            error_key = (key, slice(None, None, None))

        return self.clone(
            W_c=self.W_C[key],
            W_G=self.W_G[error_key],
            W_Gp=self.W_Gp[error_key],
        )

    def __setitem__(self, key, value: Tensor | float | int) -> None:
        if isinstance(key, tuple):
            error_key = (*key, slice(None, None, None))
        else:
            error_key = (key, slice(None, None, None))

        self.W_C[key] = value
        self.W_Gp[error_key] = value
        self.W_G[error_key] = value
