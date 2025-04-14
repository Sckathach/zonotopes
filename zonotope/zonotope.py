import gc
from typing import Optional, Tuple, Union

import torch as t
from einops import einsum
from jaxtyping import Float
from torch import Tensor

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


class Zonotope:
    """
    Implementation of the multi-norm Zonotope base object with N variables.

    The special norm is p, its dual counterpart is q. The number of special error terms is Es, and the number of infinity terms is Ei.

    Weights are stored in three matrices:
    - W_C: center (bias), shape: (N)
    - W_Ei: infinity terms, shape: (N Ei)
    - W_Es: special terms, shape: (N Es)
    """

    def __init__(
        self,
        center: Float[Tensor, "N"],
        infinity_terms: Optional[Float[Tensor, "N Ei"]] = None,
        special_terms: Optional[Float[Tensor, "N Es"]] = None,
        special_norm: int = 2,
        clone: bool = True,
    ) -> None:
        self.p = special_norm
        self.q = dual_norm(self.p)

        self.W_C: Float[Tensor, "N"] = center.clone() if clone else center

        self.device = self.W_C.device
        self.dtype = self.W_C.dtype

        if infinity_terms is None:
            self.W_Ei: Float[Tensor, "N Ei"] = t.zeros(self.N, 0)
        else:
            self.W_Ei = infinity_terms.clone() if clone else infinity_terms
        if special_terms is None:
            self.W_Es: Float[Tensor, "N Es"] = t.zeros(self.N, 0)
        else:
            self.W_Es = special_terms.clone() if clone else special_terms

    @property
    def E(self) -> int:
        return self.Es + self.Ei

    @property
    def Es(self) -> int:
        return self.W_Es.shape[-1]

    @property
    def Ei(self) -> int:
        return self.W_Ei.shape[-1]

    @property
    def N(self) -> int:
        return self.W_C.view(-1).shape[0]

    def concretize(self) -> Tuple[Float[Tensor, "N"], Float[Tensor, "N"]]:
        """Computer lower and upper bounds of the zonotope (Section 4.1)"""
        norm_infinity_terms = t.linalg.norm(self.W_Ei, ord=1, dim=-1)
        norm_special_terms = t.linalg.norm(self.W_Es, ord=self.q, dim=-1)

        lower = self.W_C - norm_infinity_terms - norm_special_terms
        upper = self.W_C + norm_infinity_terms + norm_special_terms

        return lower, upper

    def clone(self) -> "Zonotope":
        return Zonotope(
            center=self.W_C,
            infinity_terms=self.W_Ei,
            special_terms=self.W_Es,
            special_norm=self.p,
            clone=True,
        )

    def _add(self, other: Union["Zonotope", float, Tensor]) -> None:
        if isinstance(other, Zonotope):
            assert self.Ei == other.Ei
            assert self.Es == other.Es
            assert self.W_C.shape == other.W_C.shape

            self.W_C += other.W_C
            self.W_Ei += other.W_Ei
            self.W_Es += other.W_Es
        else:
            self.W_C += other
            self.W_Ei += other
            self.W_Es += other

    def add(self, other: Union["Zonotope", float, Tensor]) -> "Zonotope":
        result = self.clone()
        result._add(other)
        return result

    def _mul(self, scalar: Union[float, int, Tensor]) -> None:
        """Multiply this zonotope by a scalar in-place"""
        self.W_C *= scalar
        self.W_Ei *= scalar
        self.W_Es *= scalar

    def mul(self, scalar: Union[float, int, Tensor]) -> "Zonotope":
        result = self.clone()
        result._mul(scalar)
        return result

    def sample_point(
        self, n_samples: int = 1, binary: bool = False
    ) -> Float[Tensor, "S N"]:
        result = self.W_C.unsqueeze(0).repeat(n_samples, 1)

        if self.Es > 0:
            special_weights = t.randn(
                (n_samples, self.Es), device=self.device, dtype=self.dtype
            )
            p_norm = t.linalg.norm(special_weights, ord=self.p, dim=-1, keepdim=True)
            special_weights /= p_norm

            if not binary:
                random_scale = t.rand(n_samples, device=self.device, dtype=self.dtype)
                special_weights = einsum(
                    special_weights, random_scale, "s ei, s -> s ei"
                )

            result += einsum(self.W_Es, special_weights, "N Es, s Es -> s N")

        if self.Ei > 0:
            if binary:
                infinity_weights = (
                    t.randint(
                        0,
                        2,
                        (n_samples, self.Ei),
                        device=self.device,
                        dtype=self.dtype,
                    )
                    * 2
                    - 1
                )

            else:
                infinity_weights = (
                    t.rand((n_samples, self.Ei), device=self.device, dtype=self.dtype)
                    * 2
                    - 1
                )

            result += einsum(self.W_Ei, infinity_weights, "N Ei, s Ei -> s N")

        return result

    def remove_infinity_errors(self, n_to_keep: int) -> None:
        """Noise symbol reduction (Section 5.1)"""
        ranked_indices = self.W_Ei.abs().sum(dim=0).topk(self.Ei).indices

        self.W_Ei = t.cat(
            [
                self.W_Ei[:, ranked_indices[:n_to_keep]],
                self.W_Ei[:, ranked_indices[n_to_keep:]].sum(dim=-1, keepdim=True),
            ],
            dim=-1,
        )

    def __str__(self) -> str:
        lower, upper = self.concretize()
        difference = upper - lower
        return f"""Zonotope:
        Mean abs, lower: {lower.abs().mean().item():.5f}, upper: {upper.abs().mean().item():.5f}
        Difference, min: {difference.min().item():.5f}, max: {difference.max().item():.5f}, mean: {difference.mean().item():.5f}
        """

    __add__ = add
    __radd__ = add
    __mul__ = mul
    __rmul__ = mul
