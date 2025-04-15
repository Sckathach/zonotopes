from typing import Optional, Tuple, Union

import torch as t
from einops import einsum
from jaxtyping import Float
from torch import Tensor

from zonotope.utils import dual_norm


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
        center: Float[Tensor, "..."],
        infinity_terms: Optional[Float[Tensor, "... Ei"]] = None,
        special_terms: Optional[Float[Tensor, "... Es"]] = None,
        special_norm: int = 2,
        clone: bool = True,
    ) -> None:
        self.p = special_norm
        self.q = dual_norm(self.p)

        self.W_C: Float[Tensor, "..."] = center.clone() if clone else center

        if infinity_terms is None:
            self.W_Ei: Float[Tensor, "... Ei"] = t.zeros(
                *self.shape, 0, dtype=self.dtype, device=self.device
            )
        else:
            self.W_Ei = infinity_terms.clone() if clone else infinity_terms
        if special_terms is None:
            self.W_Es: Float[Tensor, "... Es"] = t.zeros(
                *self.shape, 0, dtype=self.dtype, device=self.device
            )
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

    @property
    def shape(self) -> t.Size:
        return self.W_C.shape

    @property
    def device(self) -> t.device:
        return self.W_C.device

    @property
    def dtype(self) -> t.dtype:
        return self.W_C.dtype

    def concretize(self) -> Tuple[Float[Tensor, "..."], Float[Tensor, "..."]]:
        """Computer lower and upper bounds of the zonotope (Section 4.1)"""
        self.update_zeros()
        norm_infinity_terms = t.linalg.norm(self.W_Ei, ord=1, dim=-1)
        norm_special_terms = t.linalg.norm(self.W_Es, ord=self.q, dim=-1)

        lower = self.W_C - norm_infinity_terms - norm_special_terms
        upper = self.W_C + norm_infinity_terms + norm_special_terms

        return lower, upper

    def update_zeros(self) -> None:
        if self.Es == 0:
            self.W_Es = t.zeros(*self.shape, 0, dtype=self.dtype, device=self.device)
        if self.Ei == 0:
            self.W_Ei = t.zeros(*self.shape, 0, dtype=self.dtype, device=self.device)

    def clone(self) -> "Zonotope":
        return Zonotope(
            center=self.W_C,
            infinity_terms=self.W_Ei,
            special_terms=self.W_Es,
            special_norm=self.p,
            clone=True,
        )

    def _add(self, other: Union["Zonotope", float, Tensor]) -> None:
        self.update_zeros()
        if isinstance(other, Zonotope):
            other.update_zeros()
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
    ) -> Float[Tensor, "S ..."]:
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
                    special_weights, random_scale, "S Ei, S -> S Ei"
                )

            result += einsum(self.W_Es, special_weights, "... Es, S Es -> S ...")

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

            result += einsum(self.W_Ei, infinity_weights, "... Ei, S Ei -> S ...")

        return result

    __add__ = add
    __radd__ = add
    __mul__ = mul
    __rmul__ = mul
