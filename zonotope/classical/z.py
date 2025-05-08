import logging
import textwrap
from typing import Any, Optional, Tuple, Union

import einops
import torch as t
from jaxtyping import Float
from torch import Tensor

from zonotope.utils import (
    dual_norm,
    get_dim_for_error_terms,
    get_einops_pattern_for_error_terms,
)

logger = logging.getLogger(__name__)


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
        """
        Initialize a zonotope.

        Args:
            center: Center point of the zonotope (bias)
            infinity_terms: Coefficient matrix for infinity-norm error terms
            special_terms: Coefficient matrix for special (p-norm) error terms
            special_norm: The p value for the special error terms (default: 2)
            clone: Whether to clone the input tensors (default: True)
        """
        self.p = special_norm
        self.q = dual_norm(self.p)

        self.W_C: Float[Tensor, "..."] = center.clone() if clone else center

        if (
            infinity_terms is None or infinity_terms.shape[-1] == 0
        ):  # second condition to reset zeros'shape
            self.W_Ei: Float[Tensor, "... Ei"] = t.zeros(
                *self.shape, 0, dtype=self.dtype, device=self.device
            )
        else:
            self.W_Ei = infinity_terms.clone() if clone else infinity_terms
        if special_terms is None or special_terms.shape[-1] == 0:
            self.W_Es: Float[Tensor, "... Es"] = t.zeros(
                *self.shape, 0, dtype=self.dtype, device=self.device
            )
        else:
            self.W_Es = special_terms.clone() if clone else special_terms

    @property
    def E(self) -> int:
        """Total number of error terms (special + infinity)."""
        return self.Es + self.Ei

    @property
    def Es(self) -> int:
        """Number of special (p-norm) error terms."""
        return self.W_Es.shape[-1]

    @property
    def Ei(self) -> int:
        """Number of infinity-norm error terms."""
        return self.W_Ei.shape[-1]

    @property
    def N(self) -> int:
        """Total number of variables in the zonotope."""
        return self.W_C.reshape(-1).shape[0]

    @property
    def shape(self) -> t.Size:
        """Shape of the center tensor."""
        return self.W_C.shape

    @property
    def device(self) -> t.device:
        """Device of the tensors."""
        return self.W_C.device

    @property
    def dtype(self) -> t.dtype:
        """Data type of the tensors."""
        return self.W_C.dtype

    @classmethod
    def from_values(
        cls,
        center: Any,
        infinity_terms: Any = None,
        special_terms: Any = None,
        special_norm: int = 2,
    ) -> "Zonotope":
        """
        Create a zonotope from various input types, converting to tensors as needed.

        Args:
            center: Center point values (will be converted to tensor)
            infinity_terms: Values for infinity-norm error terms
            special_terms: Values for special p-norm error terms
            special_norm: The p value for special error terms

        Returns:
            A new Zonotope instance
        """
        center = t.as_tensor(center, dtype=t.float)

        infinity_tensor = None
        if infinity_terms is not None:
            infinity_tensor = t.as_tensor(infinity_terms, dtype=t.float)

        special_tensor = None
        if special_terms is not None:
            special_tensor = t.as_tensor(special_terms, dtype=t.float)

        return cls(
            center=center,
            infinity_terms=infinity_tensor,
            special_terms=special_tensor,
            special_norm=special_norm,
        )

    @classmethod
    def from_bounds(cls, lower: Any, upper: Any, special_norm: int = 2) -> "Zonotope":
        """
        Create a zonotope from lower and upper bounds.

        Args:
            lower: Lower bounds
            upper: Upper bounds
            special_norm: The p value for special error terms

        Returns:
            A new Zonotope instance
        """
        center = (lower + upper) / 2
        radius = (upper - lower) / 2
        return cls.from_values(
            center=center,
            infinity_terms=radius.unsqueeze(-1),
            special_norm=special_norm,
        )

    def concretize(self) -> Tuple[Float[Tensor, "..."], Float[Tensor, "..."]]:
        """Computer lower and upper bounds of the zonotope (Section 4.1)"""
        norm_infinity_terms = t.linalg.norm(self.W_Ei, ord=1, dim=-1)
        norm_special_terms = t.linalg.norm(self.W_Es, ord=self.q, dim=-1)

        lower = self.W_C - norm_infinity_terms - norm_special_terms
        upper = self.W_C + norm_infinity_terms + norm_special_terms

        return lower, upper

    def expand_infinity_error_terms(self, n_infinity_terms: int) -> None:
        """
        Expand the number of infinity error terms to the specified value.

        ! Will be updated with a better strategy soon
        """
        if self.Ei < n_infinity_terms:
            new_terms = t.zeros(
                *self.shape,
                n_infinity_terms - self.Ei,
                dtype=self.dtype,
                device=self.device,
            )
            self.W_Ei = t.cat([self.W_Ei, new_terms], dim=-1)

    def clone(
        self,
        center: Optional[Float[Tensor, "..."]] = None,
        infinity_terms: Optional[Float[Tensor, "... Ei"]] = None,
        special_terms: Optional[Float[Tensor, "... Es"]] = None,
        special_norm: Optional[int] = None,
    ) -> "Zonotope":
        return Zonotope(
            center=self.W_C if center is None else center,
            infinity_terms=self.W_Ei if infinity_terms is None else infinity_terms,
            special_terms=self.W_Es if special_terms is None else special_terms,
            special_norm=self.p if special_norm is None else special_norm,
            clone=True,
        )

    def add(self, other: Union["Zonotope", float, int, Tensor]) -> "Zonotope":
        if isinstance(other, Zonotope):
            assert self.Es == other.Es
            assert self.W_C.shape == other.W_C.shape

            if self.Ei != other.Ei:
                logger.warning(
                    f"[!] Warning: zonotopes have different infinity error terms: {self.Ei} / {other.Ei}"
                )
                self.expand_infinity_error_terms(other.Ei)
                other.expand_infinity_error_terms(self.Ei)

            return self.clone(
                center=self.W_C + other.W_C,
                infinity_terms=self.W_Ei + other.W_Ei,
                special_terms=self.W_Es + other.W_Es,
            )

        return self.clone(center=self.W_C + other)

    def mul(self, other: Union[float, int, Tensor]) -> "Zonotope":
        """Multiply this zonotope by a scalar in-place"""
        if isinstance(other, Tensor):
            return self.clone(
                center=self.W_C * other,
                infinity_terms=self.W_Ei * other.unsqueeze(-1),
                special_terms=self.W_Es * other.unsqueeze(-1),
            )
        else:
            return self.clone(
                center=self.W_C * other,
                infinity_terms=self.W_Ei * other,
                special_terms=self.W_Es * other,
            )

    def div(self, other: Union[float, int, Tensor]) -> "Zonotope":
        return self * (1 / other)

    def sub(self, other: Union["Zonotope", float, int, Tensor]) -> "Zonotope":
        return self + (-1 * other)

    def rsub(self, other: Union["Zonotope", float, int, Tensor]) -> "Zonotope":
        return other + (-1 * self)

    @t.no_grad()
    def sample_point(
        self,
        n_samples: int = 1,
        use_binary_weights: bool = False,
        include_special_terms: bool = True,
        include_infinity_terms: bool = True,
    ) -> Float[Tensor, "S ..."]:
        """
        Sample points from within the zonotope.

        Args:
            n_samples: Number of points to sample
            use_binary_weights: Whether to use binary weights (corners of the zonotope)
            include_special_terms: Whether to include special error terms
            include_infinity_terms: Whether to include infinity error terms

        Returns:
            Tensor of sampled points with shape (n_samples, *self.shape)
        """
        result = self.W_C.unsqueeze(0).repeat(n_samples, 1)

        if self.Es > 0 and include_special_terms:
            special_weights = t.randn(
                (n_samples, self.Es), device=self.device, dtype=self.dtype
            )
            p_norm = t.linalg.norm(special_weights, ord=self.p, dim=-1, keepdim=True)
            special_weights /= p_norm

            if not use_binary_weights:
                random_scale = t.rand(n_samples, device=self.device, dtype=self.dtype)
                special_weights = einops.einsum(
                    special_weights, random_scale, "S Ei, S -> S Ei"
                )

            result += einops.einsum(self.W_Es, special_weights, "... Es, S Es -> S ...")

        if self.Ei > 0 and include_infinity_terms:
            if use_binary_weights:
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

            result += einops.einsum(
                self.W_Ei, infinity_weights, "... Ei, S Ei -> S ..."
            )

        return result

    def rearrange(self, pattern: str, **kwargs) -> "Zonotope":
        """Einops rearrange"""
        error_pattern = get_einops_pattern_for_error_terms(pattern)
        return self.clone(
            center=einops.rearrange(self.W_C, pattern, **kwargs),
            infinity_terms=einops.rearrange(self.W_Ei, error_pattern, **kwargs),
            special_terms=einops.rearrange(self.W_Es, error_pattern, **kwargs),
        )

    def repeat(self, pattern: str, **kwargs) -> "Zonotope":
        """Einops repeat"""
        error_pattern = get_einops_pattern_for_error_terms(pattern)
        return self.clone(
            center=einops.repeat(self.W_C, pattern, **kwargs),
            infinity_terms=einops.repeat(self.W_Ei, error_pattern, **kwargs),
            special_terms=einops.repeat(self.W_Es, error_pattern, **kwargs),
        )

    def einsum(self, other: Tensor, pattern: str, **kwargs) -> "Zonotope":
        """Einops einsum"""
        error_pattern = get_einops_pattern_for_error_terms(pattern)
        return self.clone(
            center=einops.einsum(self.W_C, other, pattern, **kwargs),
            infinity_terms=einops.einsum(self.W_Ei, other, error_pattern, **kwargs),
            special_terms=einops.einsum(self.W_Es, other, error_pattern, **kwargs),
        )

    def sum(self, dim: int, **kwargs) -> "Zonotope":
        """Torch sum"""
        error_dim = get_dim_for_error_terms(dim)
        return self.clone(
            center=self.W_C.sum(dim=dim, **kwargs),
            infinity_terms=self.W_Ei.sum(dim=error_dim, **kwargs)
            if self.Ei > 0
            else None,
            special_terms=self.W_Es.sum(dim=error_dim, **kwargs)
            if self.Es > 0
            else None,
        )

    def to(
        self, device: Optional[t.device] = None, dtype: Optional[t.dtype] = None
    ) -> "Zonotope":
        """Torch to"""
        device = self.device if device is None else device
        dtype = self.dtype if dtype is None else dtype
        return self.clone(
            center=self.W_C.to(device=device, dtype=dtype),
            infinity_terms=self.W_Ei.to(device=device, dtype=dtype),
            special_terms=self.W_Es.to(device=device, dtype=dtype),
        )

    def contiguous(self) -> None:
        """Torch contiguous"""
        self.W_C.contiguous()
        self.W_Ei.contiguous()
        self.W_Es.contiguous()

    def __getitem__(self, key):
        if isinstance(key, tuple):
            error_key = (*key, slice(None, None, None))
        else:
            error_key = (key, slice(None, None, None))

        return self.clone(
            center=self.W_C[key],
            infinity_terms=self.W_Ei[error_key],
            special_terms=self.W_Es[error_key],
        )

    def __setitem__(self, key, value: Tensor | float | int) -> None:
        #! not tested
        if isinstance(key, tuple):
            error_key = (*key, slice(None, None, None))
        else:
            error_key = (key, slice(None, None, None))

        self.W_C[key] = value
        self.W_Ei[error_key] = value
        self.W_Es[error_key] = value

    def __repr__(self) -> str:
        lower, upper = self.concretize()
        return textwrap.dedent(f"""
            ---
            Center: 
                {self.W_C}
        {
            f'''
            Ei: 
                {self.W_Ei}

            shape: {self.W_Ei.shape}
            '''
            if self.Ei > 0
            else ""
        }
        {
            f'''
            Es: 
                {self.W_Es}

            shape: {self.W_Es.shape}
            '''
            if self.Es > 0
            else ""
        }

            Lower: 
                {lower}

            Upper: 
                {upper}
            ---
        """)

    def __len__(self) -> int:
        return self.N

    __add__ = add
    __radd__ = add
    __mul__ = mul
    __rmul__ = mul
    __sub__ = sub
    __rsub__ = rsub
    __div__ = div
