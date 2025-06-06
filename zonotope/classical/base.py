import math
from types import EllipsisType
from typing import Any, List, Optional, Self

import torch as t
from jaxtyping import Float
from torch import Tensor


class ZonotopeBase:
    W_C: Float[Tensor, "..."]
    W_G: Float[Tensor, "... I"]

    @property
    def I(self) -> int:
        return self.W_G.shape[-1]

    @property
    def N(self) -> int:
        return math.prod(self.W_C.shape)

    @property
    def shape(self) -> t.Size:
        return self.W_C.shape

    @property
    def device(self) -> t.device:
        """Device of the tensors."""
        return self.W_C.device

    @property
    def dtype(self) -> t.dtype:
        """Data type of the tensors."""
        return self.W_C.dtype

    def zeros(self, *shape, **kwargs) -> Float[Tensor, "..."]:
        kwargs = {"device": self.device, "dtype": self.dtype} | kwargs
        return t.zeros(*shape, **kwargs)  # type: ignore

    def ones(self, *shape, **kwargs) -> Float[Tensor, "..."]:
        kwargs = {"device": self.device, "dtype": self.dtype} | kwargs
        return t.ones(*shape, **kwargs)  # type: ignore

    def eye(self, *shape, **kwargs) -> Float[Tensor, "..."]:
        kwargs = {"device": self.device, "dtype": self.dtype} | kwargs
        return t.eye(*shape, **kwargs)  # type: ignore

    def as_tensor(self, obj: Any) -> Tensor:
        return t.as_tensor(obj, dtype=self.dtype, device=self.device)

    def as_sparse_tensor(self, obj: Any) -> Tensor:
        return t.as_tensor(obj, dtype=self.dtype, device=self.device).to_sparse_coo()

    def add(self, other: Self | float | int | Tensor) -> Self:
        raise NotImplementedError

    def mul(self, other: float | int | Tensor) -> Self:
        raise NotImplementedError

    def sub(self, other: Self | float | int | Tensor) -> Self:
        return self + (-1 * other)

    def rsub(self, other: Self | float | int | Tensor) -> Self:
        return other + (-1 * self)

    def div(self, other: float | int | Tensor) -> Self:
        return self * (1 / other)

    def cat(
        self,
        *elements: List[Tensor | tuple],
        row_dims: Optional[EllipsisType | int] = None,
        column_dims: Optional[EllipsisType | int] = None,
    ) -> Tensor:
        if len(elements) == 0:
            raise ValueError("No elements provided in self.cat")

        def create_zeros(shape):
            return (
                self.zeros(*shape).to_sparse_coo()
                if self.W_G.is_sparse
                else self.zeros(*shape)
            )

        return t.cat(
            [
                t.cat(
                    [create_zeros(j) if isinstance(j, tuple) else j for j in i],
                    dim=column_dims if column_dims is not None else -1,
                )
                for i in elements
            ],
            dim=row_dims if row_dims is not None else 0,
        )

    def __len__(self) -> int:
        return self.N

    __add__ = add
    __radd__ = add
    __mul__ = mul
    __rmul__ = mul
    __sub__ = sub
    __rsub__ = rsub
    __div__ = div
