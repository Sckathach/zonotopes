import textwrap
from types import EllipsisType
from typing import Any, List, Optional, Self

import torch as t
from jaxtyping import Float
from pydantic import BaseModel, PositiveInt
from torch import Tensor

from zonotope import DEFAULT_DEVICE, DEFAULT_DTYPE
from zonotope.hcz import DEFAULT_LR, DEFAULT_N_STEPS


class HCZConfig(BaseModel):
    lr: float = DEFAULT_LR
    n_steps: PositiveInt = DEFAULT_N_STEPS


class HCZBase:
    W_C: Tensor
    W_G: Tensor
    W_Gp: Tensor
    W_A: Tensor
    W_Ap: Tensor
    W_B: Tensor
    config: HCZConfig

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
        self.config = config if config is not None else HCZConfig(**kwargs)
        self.W_C: Float[Tensor, "..."] = W_C.clone() if clone else W_C

        if W_G is None or W_G.shape[0] == 0:  # second condition to reset zeros'shape
            self.W_G: Float[Tensor, "I ..."] = self.zeros(0, *self.shape)
        else:
            self.W_G = W_G.clone() if clone else W_G

        if W_Gp is None or W_Gp.shape[0] == 0:
            self.W_Gp: Float[Tensor, "Ip ..."] = self.zeros(0, *self.shape)
        else:
            self.W_Gp = W_Gp.clone() if clone else W_Gp

        if W_B is None or W_B.shape[0] == 0:
            self.W_B: Float[Tensor, "J"] = self.zeros(0)
        else:
            self.W_B = W_B.clone() if clone else W_B

        if W_A is None or W_A.shape[0] == 0:
            self.W_A: Float[Tensor, "I J"] = self.zeros(self.I, self.J)
        else:
            self.W_A = W_A.clone() if clone else W_A

        if W_Ap is None or W_Ap.shape[0] == 0:
            self.W_Ap: Float[Tensor, "Ip J"] = self.zeros(self.Ip, self.J)
        else:
            self.W_Ap = W_Ap.clone() if clone else W_Ap

    @property
    def J(self) -> int:
        return self.W_B.shape[0]

    @property
    def Ip(self) -> int:
        return self.W_Gp.shape[0]

    @property
    def I(self) -> int:
        return self.W_G.shape[0]

    @property
    def N(self) -> int:
        return self.W_C.shape[0]

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

    def display_shapes(self) -> None:
        print(
            textwrap.dedent(f"""
                c: {self.W_C.shape}
                G: {self.W_G.shape}
                G': {self.W_Gp.shape}
                A: {self.W_A.shape}
                A': {self.W_Ap.shape}
                b: {self.W_B.shape}
            """)
        )

    def display_weights(self) -> str:
        return textwrap.dedent(f"""
            c: {self.W_C}
            G: {self.W_G}
            G': {self.W_Gp}
            A: {self.W_A}
            A': {self.W_Ap}
            b: {self.W_B}
        """)

    def load_config_from_(self, other: Self) -> None:
        self.config = other.config
        self.to(device=other.device, dtype=other.dtype)

    def to(
        self,
        device: Optional[t.device] = None,
        dtype: Optional[t.dtype] = None,
    ) -> Self:
        """Torch to"""
        device = self.device if device is None else device
        dtype = self.dtype if dtype is None else dtype

        return self.clone(
            W_C=self.W_C.to(device=device, dtype=dtype),
            W_G=self.W_G.to(device=device, dtype=dtype),
            W_Gp=self.W_Gp.to(device=device, dtype=dtype),
            W_A=self.W_A.to(device=device, dtype=dtype),
            W_Ap=self.W_Ap.to(device=device, dtype=dtype),
            W_B=self.W_B.to(device=self.device, dtype=self.dtype),
        )

    def clone(
        self,
        W_C: Optional[Float[Tensor, "..."]] = None,
        W_G: Optional[Float[Tensor, "I ..."]] = None,
        W_Gp: Optional[Float[Tensor, "Ip ..."]] = None,
        W_A: Optional[Float[Tensor, "I J"]] = None,
        W_Ap: Optional[Float[Tensor, "Ip J"]] = None,
        W_B: Optional[Float[Tensor, "J"]] = None,
        config: Optional[HCZConfig] = None,
        **kwargs,
    ) -> Self:
        result = self.__class__(
            W_C=self.W_C if W_C is None else W_C,
            W_G=self.W_G if W_G is None else W_G,
            W_Gp=self.W_Gp if W_Gp is None else W_Gp,
            W_A=self.W_A if W_A is None else W_A,
            W_Ap=self.W_Ap if W_Ap is None else W_Ap,
            W_B=self.W_B if W_B is None else W_B,
            config=self.config if config is None else config,
            **kwargs,
        )
        result.check_integrity()

        return result

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
        def as_tensor(obj: Any) -> Tensor:
            return t.as_tensor(obj, dtype=dtype, device=device)

        result = cls(
            W_C=as_tensor(W_C),
            W_G=as_tensor(W_G) if W_G is not None else None,
            W_Gp=as_tensor(W_Gp) if W_Gp is not None else None,
            W_A=as_tensor(W_A) if W_A is not None else None,
            W_Ap=as_tensor(W_Ap) if W_Ap is not None else None,
            W_B=as_tensor(W_B) if W_B is not None else None,
            config=config,
            **kwargs,
        )
        result.check_integrity()

        return result

    def check_integrity(self) -> None:
        assert self.W_C.device == self.device
        assert self.W_G.device == self.device
        assert self.W_Gp.device == self.device
        assert self.W_A.device == self.device
        assert self.W_Ap.device == self.device
        assert self.W_B.device == self.device
        assert self.W_C.dtype == self.dtype
        assert self.W_G.dtype == self.dtype
        assert self.W_Gp.dtype == self.dtype
        assert self.W_A.dtype == self.dtype
        assert self.W_Ap.dtype == self.dtype
        assert self.W_B.dtype == self.dtype

    @classmethod
    def empty(cls, config: Optional[HCZConfig] = None, **kwargs) -> Self:
        return cls.from_values(W_c=[], config=config, **kwargs)

    def empty_from_self(self) -> Self:
        return self.__class__.empty(
            config=self.config, device=self.device, dtype=self.dtype
        )

    def mul(self, other: float | int | Tensor) -> Self:
        raise NotImplementedError

    def add(self, other: Self | float | int | Tensor) -> Self:
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
    __repr__ = display_weights
    __str__ = display_weights
