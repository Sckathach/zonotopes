import logging
from typing import Any, Optional, Self, Tuple

import einops
import torch as t
from jaxtyping import Float
from torch import Tensor

from zonotope import DEFAULT_DEVICE, DEFAULT_DTYPE
from zonotope.classical.base import ZonotopeBase
from zonotope.utils import (
    get_dim_for_error_terms,
    get_einops_pattern_for_error_terms,
)

logger = logging.getLogger(__name__)


class Zonotope(ZonotopeBase):
    def __init__(
        self,
        W_C: Float[Tensor, "..."],
        W_G: Optional[Float[Tensor, "... I"]] = None,
        clone: bool = True,
    ) -> None:
        self.W_C: Float[Tensor, "..."] = W_C.clone() if clone else W_C

        if W_G is None or W_G.shape[-1] == 0:  # second condition to reset zeros'shape
            self.W_G: Float[Tensor, "... I"] = t.zeros(
                *self.shape, 0, dtype=self.dtype, device=self.device
            )
        else:
            self.W_G = W_G.clone() if clone else W_G

    @classmethod
    def from_values(
        cls,
        W_C: Any,
        W_G: Any = None,
        dtype: t.dtype = DEFAULT_DTYPE,
        device: t.device = DEFAULT_DEVICE,
        **kwargs,
    ) -> Self:
        def as_tensor(obj: Any) -> Tensor:
            return t.as_tensor(obj, dtype=dtype, device=device)

        result = cls(
            W_C=as_tensor(W_C),
            W_G=as_tensor(W_G) if W_G is not None else None,
            **kwargs,
        )

        return result

    @classmethod
    def from_bounds(cls, lower: Any, upper: Any) -> Self:
        center = (lower + upper) / 2
        radius = (upper - lower) / 2
        return cls.from_values(
            W_C=center,
            W_G=radius.unsqueeze(-1),
        )

    def concretize(self) -> Tuple[Float[Tensor, "..."], Float[Tensor, "..."]]:
        norm_infinity_terms = t.linalg.norm(self.W_G, ord=1, dim=-1)

        lower = self.W_C - norm_infinity_terms
        upper = self.W_C + norm_infinity_terms

        return lower, upper

    def expand_infinity_error_terms(self, new_I: int) -> None:
        """
        Expand the number of infinity error terms to the specified value.

        ! Will be updated with a better strategy soon
        """
        if new_I > self.I:
            new_terms = self.zeros(
                *self.shape,
                new_I - self.I,
            )
            self.W_G = self.cat([self.W_G, new_terms])

    def clone(
        self,
        W_C: Optional[Float[Tensor, "..."]] = None,
        W_G: Optional[Float[Tensor, "... I"]] = None,
    ) -> Self:
        return self.__class__(
            W_C=self.W_C if W_C is None else W_C,
            W_G=self.W_G if W_G is None else W_G,
            clone=True,
        )

    def add(self, other: Self | float | int | Tensor) -> Self:
        if isinstance(other, Zonotope):
            assert self.W_C.shape == other.W_C.shape

            if self.I != other.I:
                logger.warning(
                    f"[!] Warning: zonotopes have different infinity error terms: {self.I} / {other.I}"
                )
                self.expand_infinity_error_terms(other.I)
                other.expand_infinity_error_terms(self.I)

            return self.clone(
                W_C=self.W_C + other.W_C,
                W_G=self.W_G + other.W_G,
            )

        return self.clone(W_C=self.W_C + other)

    def mul(self, other: float | int | Tensor) -> Self:
        """Multiply this zonotope by a scalar in-place"""
        if isinstance(other, Tensor):
            return self.clone(
                W_C=self.W_C * other,
                W_G=self.W_G * other.unsqueeze(-1),
            )
        else:
            return self.clone(
                W_C=self.W_C * other,
                W_G=self.W_G * other,
            )

    @t.no_grad()
    def sample_point(
        self,
        n_samples: int = 1,
        use_binary_weights: bool = False,
    ) -> Float[Tensor, "S ..."]:
        """
        Sample points from within the zonotope.

        Args:
            n_samples: Number of points to sample
            use_binary_weights: Whether to use binary weights (corners of the zonotope)

        Returns:
            Tensor of sampled points with shape (n_samples, *self.shape)
        """
        result = self.W_C.unsqueeze(0).repeat(n_samples, 1)

        if self.I > 0:
            if use_binary_weights:
                infinity_weights = (
                    t.randint(
                        0,
                        2,
                        (n_samples, self.I),
                        device=self.device,
                        dtype=self.dtype,
                    )
                    * 2
                    - 1
                )

            else:
                infinity_weights = (
                    t.rand((n_samples, self.I), device=self.device, dtype=self.dtype)
                    * 2
                    - 1
                )

            result += einops.einsum(self.W_G, infinity_weights, "... I, S I -> S ...")

        return result

    def rearrange(self, pattern: str, **kwargs) -> Self:
        """Einops rearrange"""
        error_pattern = get_einops_pattern_for_error_terms(pattern)
        return self.clone(
            W_C=einops.rearrange(self.W_C, pattern, **kwargs),
            W_G=einops.rearrange(self.W_G, error_pattern, **kwargs),
        )

    def repeat(self, pattern: str, **kwargs) -> Self:
        """Einops repeat"""
        error_pattern = get_einops_pattern_for_error_terms(pattern)
        return self.clone(
            W_C=einops.repeat(self.W_C, pattern, **kwargs),
            W_G=einops.repeat(self.W_G, error_pattern, **kwargs),
        )

    def einsum(self, other: Tensor, pattern: str, **kwargs) -> Self:
        """Einops einsum"""
        error_pattern = get_einops_pattern_for_error_terms(pattern)
        return self.clone(
            W_C=einops.einsum(self.W_C, other, pattern, **kwargs),
            W_G=einops.einsum(self.W_G, other, error_pattern, **kwargs),
        )

    def sum(self, dim: int, **kwargs) -> Self:
        """Torch sum"""
        error_dim = get_dim_for_error_terms(dim)
        return self.clone(
            W_C=self.W_C.sum(dim=dim, **kwargs),
            W_G=self.W_G.sum(dim=error_dim, **kwargs) if self.I > 0 else None,
        )

    def to(
        self, device: Optional[t.device] = None, dtype: Optional[t.dtype] = None
    ) -> Self:
        """Torch to"""
        device = self.device if device is None else device
        dtype = self.dtype if dtype is None else dtype
        return self.clone(
            W_C=self.W_C.to(device=device, dtype=dtype),
            W_G=self.W_G.to(device=device, dtype=dtype),
        )

    def contiguous(self) -> None:
        """Torch contiguous"""
        self.W_C.contiguous()
        self.W_G.contiguous()

    def __getitem__(self, key):
        if isinstance(key, tuple):
            error_key = (*key, slice(None, None, None))
        else:
            error_key = (key, slice(None, None, None))

        return self.clone(
            W_C=self.W_C[key],
            W_G=self.W_G[error_key],
        )

    def __setitem__(self, key, value: Tensor | float | int) -> None:
        #! not tested
        if isinstance(key, tuple):
            error_key = (*key, slice(None, None, None))
        else:
            error_key = (key, slice(None, None, None))

        self.W_C[key] = value
        self.W_G[error_key] = value

    def assert_equal(self, W_C: Any = None, W_G: Any = None) -> None:
        if W_C is not None:
            try:
                assert t.allclose(self.W_C, self.as_tensor(W_C))
            except AssertionError as e:
                print(f"Centers are differents: \n{self.W_C}\n{self.as_tensor(W_C)}\n")
                raise AssertionError(e) from e
        if W_G is not None:
            try:
                assert t.allclose(self.W_G, self.as_tensor(W_G))
            except AssertionError as e:
                print(
                    f"Generators are differents: \n{self.W_G}\n{self.as_tensor(W_G)}\n"
                )
                raise AssertionError(e) from e

    def equal(self, other: Any) -> bool:
        if not isinstance(other, Zonotope):
            return False

        other = other.to(device=self.device, dtype=self.dtype)
        return t.allclose(self.W_C, other.W_C) and t.allclose(self.W_G, other.W_G)
