import textwrap
from typing import Any, Optional

import torch as t
from jaxtyping import Float
from pydantic import BaseModel, PositiveInt
from torch import Tensor

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

    J: int
    Ip: int
    I: int
    N: int

    def __init__(
        self,
        W_C: Float[Tensor, "N"],
        W_G: Optional[Float[Tensor, "I N"]] = None,
        W_Gp: Optional[Float[Tensor, "Ip N"]] = None,
        W_A: Optional[Float[Tensor, "I J"]] = None,
        W_Ap: Optional[Float[Tensor, "Ip J"]] = None,
        W_B: Optional[Float[Tensor, "J"]] = None,
        config: Optional[HCZConfig] = None,
        clone: bool = True,
        **kwargs,
    ) -> None: ...

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

    def load_config_from_(self, other: "HCZBase") -> None:
        self.config = other.config
        self.to(device=other.device, dtype=other.dtype)

    def to(
        self,
        device: Optional[t.device] = None,
        dtype: Optional[t.dtype] = None,
    ) -> "HCZBase":
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
        W_C: Optional[Float[Tensor, "N"]] = None,
        W_G: Optional[Float[Tensor, "I N"]] = None,
        W_Gp: Optional[Float[Tensor, "Ip N"]] = None,
        W_A: Optional[Float[Tensor, "I J"]] = None,
        W_Ap: Optional[Float[Tensor, "Ip J"]] = None,
        W_B: Optional[Float[Tensor, "J"]] = None,
        config: Optional[HCZConfig] = None,
        **kwargs,
    ) -> "HCZBase":
        result = HCZBase(
            W_C=self.W_C if W_C is None else W_C,
            W_G=self.W_G if W_G is None else W_G,
            W_Gp=self.W_Gp if W_Gp is None else W_Gp,
            W_A=self.W_A if W_A is None else W_A,
            W_Ap=self.W_Ap if W_Ap is None else W_Ap,
            W_B=self.W_B if W_B is None else W_B,
            config=self.config if config is None else config,
            **kwargs,
        )
        return result
