import textwrap
from types import EllipsisType
from typing import Any, List, Optional, Tuple, Union

import torch as t
from jaxtyping import Float
from pydantic import BaseModel, PositiveInt
from torch import Tensor
from torch.linalg import norm

from zonotope import DEFAULT_DEVICE, DEFAULT_DTYPE
from zonotope.hybrid_constrained import DEFAULT_LR, DEFAULT_N_STEPS
from zonotope.hybrid_constrained.optimise import optimize_lambda


class HCZConfig(BaseModel):
    lr: float = DEFAULT_LR
    n_steps: PositiveInt = DEFAULT_N_STEPS


class HCZ:
    def __init__(
        self,
        W_c: Float[Tensor, "N"],
        W_G: Optional[Float[Tensor, "I N"]] = None,
        W_Gp: Optional[Float[Tensor, "Ip N"]] = None,
        W_A: Optional[Float[Tensor, "I J"]] = None,
        W_Ap: Optional[Float[Tensor, "Ip J"]] = None,
        W_b: Optional[Float[Tensor, "J"]] = None,
        config: Optional[HCZConfig] = None,
        clone: bool = True,
        **kwargs,
    ) -> None:
        self.config = config if config is not None else HCZConfig(**kwargs)
        self.W_C: Float[Tensor, "N"] = W_c.clone() if clone else W_c

        if W_G is None or W_G.shape[0] == 0:  # second condition to reset zeros'shape
            self.W_G: Float[Tensor, "I N"] = self.zeros(0, self.N).to_sparse_coo()
        else:
            self.W_G = W_G.clone() if clone else W_G

        if W_Gp is None or W_Gp.shape[0] == 0:
            self.W_Gp: Float[Tensor, "Ip N"] = self.zeros(0, self.N).to_sparse_coo()
        else:
            self.W_Gp = W_Gp.clone() if clone else W_Gp

        if W_b is None or W_b.shape[0] == 0:
            self.W_B: Float[Tensor, "J"] = self.zeros(0)
        else:
            self.W_B = W_b.clone() if clone else W_b

        if W_A is None or W_A.shape[0] == 0:
            self.W_A: Float[Tensor, "I J"] = self.zeros(self.I, self.J).to_sparse_coo()
        else:
            self.W_A = W_A.clone() if clone else W_A

        if W_Ap is None or W_Ap.shape[0] == 0:
            self.W_Ap: Float[Tensor, "Ip J"] = self.zeros(
                self.Ip, self.J
            ).to_sparse_coo()
        else:
            self.W_Ap = W_Ap.clone() if clone else W_Ap

    @property
    def J(self) -> int:
        return self.W_B.shape[0]

    @property
    def Ip(self) -> int:
        return self.W_Gp.shape[0]

    @property
    def I(self) -> int:  # noqa: E743
        return self.W_G.shape[0]

    @property
    def N(self) -> int:
        return self.W_C.shape[0]

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

    def load_config_from_(self, other: "HCZ") -> None:
        self.config = other.config
        self.to(device=other.device, dtype=other.dtype)

    def check_integrity(self) -> None:
        assert not self.W_C.is_sparse
        assert not self.W_B.is_sparse
        assert self.W_G.is_sparse
        assert self.W_Gp.is_sparse
        assert self.W_A.is_sparse
        assert self.W_Ap.is_sparse
        assert self.W_G.shape == t.Size([self.I, self.N])
        assert self.W_Gp.shape == t.Size([self.Ip, self.N])
        assert self.W_A.shape == t.Size([self.I, self.J])
        assert self.W_Ap.shape == t.Size([self.Ip, self.J])
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
    def from_values(
        cls,
        W_c: Any,
        W_G: Any = None,
        W_Gp: Any = None,
        W_A: Any = None,
        W_Ap: Any = None,
        W_b: Any = None,
        dtype: t.dtype = DEFAULT_DTYPE,
        device: t.device = DEFAULT_DEVICE,
        config: Optional[HCZConfig] = None,
        **kwargs,
    ) -> "HCZ":
        def as_tensor(obj: Any) -> Tensor:
            return t.as_tensor(obj, dtype=dtype, device=device)

        def as_sparse_tensor(obj: Any) -> Tensor:
            return t.as_tensor(obj, dtype=dtype, device=device).to_sparse_coo()

        result = cls(
            W_c=t.as_tensor(W_c, dtype=dtype, device=device),
            W_G=as_sparse_tensor(W_G) if W_G is not None else None,
            W_Gp=as_sparse_tensor(W_Gp) if W_Gp is not None else None,
            W_A=as_sparse_tensor(W_A) if W_A is not None else None,
            W_Ap=as_sparse_tensor(W_Ap) if W_Ap is not None else None,
            W_b=as_tensor(W_b) if W_b is not None else None,
            config=config,
            **kwargs,
        )
        result.check_integrity()
        return result

    @classmethod
    def from_bounds(
        cls,
        lower: Any,
        upper: Any,
        dtype: t.dtype = DEFAULT_DTYPE,
        device: t.device = DEFAULT_DEVICE,
        config: Optional[HCZConfig] = None,
        **kwargs,
    ) -> "HCZ":
        lower_ = t.as_tensor(lower, dtype=dtype, device=device)
        upper_ = t.as_tensor(upper, dtype=dtype, device=device)

        center = (lower_ + upper_) / 2
        radius = (upper_ - lower_) / 2
        mask = radius != 0
        N = center.shape[0]
        I = int(mask.sum().item())  # noqa: E741
        radius_non_zeros = t.eye(I, dtype=dtype, device=device) * radius[mask]
        radius_expanded = t.zeros(I, N, dtype=dtype, device=device)
        radius_expanded[:, mask] = radius_non_zeros
        return cls.from_values(W_c=center, W_G=radius_expanded, config=config, **kwargs)

    @classmethod
    def empty(cls, config: Optional[HCZConfig] = None, **kwargs) -> "HCZ":
        return cls.from_values(W_c=[], config=config, **kwargs)

    def empty_from_self(self) -> "HCZ":
        return HCZ.empty(config=self.config, device=self.device, dtype=self.dtype)

    def clone(
        self,
        W_c: Optional[Float[Tensor, "N"]] = None,
        W_G: Optional[Float[Tensor, "I N"]] = None,
        W_Gp: Optional[Float[Tensor, "Ip N"]] = None,
        W_A: Optional[Float[Tensor, "I J"]] = None,
        W_Ap: Optional[Float[Tensor, "Ip J"]] = None,
        W_b: Optional[Float[Tensor, "J"]] = None,
        config: Optional[HCZConfig] = None,
    ) -> "HCZ":
        result = HCZ(
            W_c=self.W_C if W_c is None else W_c,
            W_G=self.W_G if W_G is None else W_G,
            W_Gp=self.W_Gp if W_Gp is None else W_Gp,
            W_A=self.W_A if W_A is None else W_A,
            W_Ap=self.W_Ap if W_Ap is None else W_Ap,
            W_b=self.W_B if W_b is None else W_b,
            config=self.config if config is None else config,
        )
        result.check_integrity()
        return result

    def concretize(self, **kwargs) -> Tuple[Float[Tensor, "N"], Float[Tensor, "N"]]:
        """
        Returns: lower, upper
        """
        if self.J == 0:
            return self.dual_lower(self.zeros(self.J, self.N)), -self.dual_upper(
                self.zeros(self.J, self.N)
            )

        kwargs = {"lr": self.config.lr, "n_steps": self.config.n_steps} | kwargs
        lambda_lower = optimize_lambda(
            (self.J, self.N),
            self.dual_lower,
            device=self.device,
            dtype=self.dtype,
            **kwargs,
        )
        lambda_upper = optimize_lambda(
            (self.J, self.N),
            self.dual_upper,
            device=self.device,
            dtype=self.dtype,
            **kwargs,
        )
        lower, upper = self.dual_lower(lambda_lower), -self.dual_upper(lambda_upper)
        lower_0, upper_0 = (
            self.dual_lower(self.zeros(self.J, self.N)),
            -self.dual_upper(self.zeros(self.J, self.N)),
        )
        mask_lower = lower_0 > lower
        mask_upper = upper_0 < upper
        lambda_lower[:, mask_lower] = 0
        lambda_upper[:, mask_upper] = 0

        return self.dual_lower(lambda_lower), -self.dual_upper(lambda_upper)

    def dual_lower(self, lmda: Float[Tensor, "J N"]) -> Float[Tensor, "N"]:
        return (
            -norm(
                -t.sparse.mm(self.W_A, lmda) + self.W_G,  # I J, J N -> I N
                ord=1,
                dim=0,
            )
            - norm(
                -t.sparse.mm(self.W_Ap, lmda) + self.W_Gp,  # Ip J, J N -> Ip N
                ord=1,
                dim=0,
            )
            + self.W_C
            + self.W_B @ lmda  # J, J N -> N
        )

    def dual_upper(self, lmda: Float[Tensor, "J N"]) -> Float[Tensor, "N"]:
        return (
            -norm(
                -t.sparse.mm(self.W_A, lmda) - self.W_G,  # I J, J N -> I
                ord=1,
                dim=0,
            )
            - norm(
                -t.sparse.mm(self.W_Ap, lmda) - self.W_Gp,  # Ip J, J N -> N Ip
                ord=1,
                dim=0,
            )
            - self.W_C
            + self.W_B @ lmda  # J, J N -> N
        )

    def add(self, other: Union["HCZ", float, int, Tensor]) -> "HCZ":
        if isinstance(other, HCZ):
            if self.N == 0:
                return other
            if other.N == 0:
                return self

            return self.clone(
                W_c=self.W_C + other.W_C,
                W_G=self.cat([self.W_G], [other.W_G]),  # I1 + I2, N
                W_Gp=self.cat([self.W_Gp], [other.W_Gp]),  # Ip1 + Ip2, N
                W_A=self.cat(
                    [self.W_A, (self.I, other.J)],
                    [(other.I, self.J), other.W_A],
                ),
                W_Ap=self.cat(
                    [self.W_Ap, (self.Ip, other.J)], [(other.Ip, self.J), other.W_Ap]
                ),
                W_b=self.cat([self.W_B, other.W_B]),
            )

        return self.clone(W_c=self.W_C + other)

    def mul(self, other: Union[float, int, Tensor]) -> "HCZ":
        if isinstance(other, Tensor):
            return self.clone(
                W_c=self.W_C * other,
                W_G=self.W_G * other.unsqueeze(0),
                W_Gp=self.W_Gp * other.unsqueeze(0),
            )
        else:
            return self.clone(
                W_c=self.W_C * other,
                W_G=self.W_G * other,
                W_Gp=self.W_Gp * other,
            )

    def mm(
        self, other: Float[Tensor, "N M"], check_emptiness: bool = True, **kwargs
    ) -> "HCZ":
        if check_emptiness and self.is_empty(**kwargs):
            return self.empty_from_self()

        return self.clone(
            W_c=self.W_C @ other,
            W_G=t.sparse.mm(self.W_G, other.to_sparse_coo()),
            W_Gp=t.sparse.mm(self.W_Gp, other.to_sparse_coo()),
        )

    def sub(self, other: Union["HCZ", float, int, Tensor]) -> "HCZ":
        return self + (-1 * other)

    def rsub(self, other: Union["HCZ", float, int, Tensor]) -> "HCZ":
        return other + (-1 * self)

    def div(self, other: Union[float, int, Tensor]) -> "HCZ":
        return self * (1 / other)

    def intersect(
        self,
        other: "HCZ",
        r: Optional[Float[Tensor, "N1 N2"]] = None,
        check_emptiness_before: bool = True,
        check_emptiness_after: bool = True,
        **kwargs,
    ) -> "HCZ":
        if check_emptiness_before and (
            self.is_empty(**kwargs) or other.is_empty(**kwargs) == 0
        ):
            return self.empty_from_self()
        if r is not None:
            rg = t.sparse.mm(self.W_G, r.to_sparse_coo())
            rgp = t.sparse.mm(self.W_Gp, r.to_sparse_coo())
            rc = self.W_C @ r
        else:
            rg, rgp, rc = self.W_G, self.W_Gp, self.W_C

        result = self.clone(
            W_G=self.cat([self.W_G], [(other.I, self.N)]),
            W_Gp=self.cat([self.W_Gp], [(other.Ip, self.N)]),
            W_A=self.cat(
                [self.W_A, (self.I, other.J), rg],
                [(other.I, self.J), other.W_A, -other.W_G],
            ),
            W_Ap=self.cat(
                [self.W_Ap, (self.Ip, other.J), rgp],
                [(other.Ip, self.J), other.W_Ap, -other.W_Gp],
            ),
            W_b=self.cat([self.W_B, other.W_B, other.W_C - rc]),
        )
        if check_emptiness_after and result.is_empty(**kwargs):
            return self.empty_from_self()

        return result

    def is_empty(self, **kwargs) -> bool:
        if self.N == 0:
            return True

        lower, upper = self.concretize(**kwargs)
        return bool(t.any(lower > upper).item())

    def cartesian_product(
        self, other: "HCZ", check_emptiness: bool = True, **kwargs
    ) -> "HCZ":
        if check_emptiness:
            if self.is_empty(**kwargs):
                return other
            if other.is_empty(**kwargs):
                return self

        return self.clone(
            W_c=self.cat([self.W_C, other.W_C]),
            W_G=self.cat([self.W_G, (self.I, other.N)], [(other.I, self.N), other.W_G]),
            W_Gp=self.cat(
                [self.W_Gp, (self.Ip, other.N)], [(other.Ip, self.N), other.W_Gp]
            ),
            W_A=self.cat([self.W_A, (self.I, other.J)], [(other.I, self.J), other.W_A]),
            W_Ap=self.cat(
                [self.W_Ap, (self.Ip, other.J)], [(other.Ip, self.J), other.W_Ap]
            ),
            W_b=self.cat([self.W_B, other.W_B]),
        )

    def union(self, other: "HCZ", check_emptiness: bool = True, **kwargs) -> "HCZ":
        if check_emptiness:
            if self.is_empty(**kwargs):
                return other
            if other.is_empty(**kwargs):
                return self

        I1, I2, Ip1, Ip2, J1, J2 = self.I, other.I, self.Ip, other.Ip, self.J, other.J
        Inew = 2 * I1 + 2 * Ip1 + 2 * I2 + 2 * Ip2

        def seye(n: int) -> Tensor:
            return self.eye(n).to_sparse_coo()

        def sones(*shape) -> Tensor:
            return self.ones(*shape).to_sparse_coo()

        return self.clone(
            W_c=1
            / 2
            * (
                self.W_C
                + other.W_C
                + self.W_Gp.sum(0).to_dense()
                + other.W_Gp.sum(0).to_dense()
            ),
            W_G=self.cat([self.W_G], [other.W_G], [(Inew, self.N)]),
            W_Gp=self.cat(
                [self.W_Gp],
                [other.W_Gp],
                [
                    1
                    / 2
                    * (
                        self.W_C.to_sparse_coo()
                        - other.W_C.to_sparse_coo()
                        + self.W_Gp.sum(0)
                        - other.W_Gp.sum(0)
                    ).unsqueeze(0),
                ],
            ),
            W_A=self.cat(
                [self.W_A, (I1, J2), seye(I1), -seye(I1), (I1, Inew - 2 * I1)],
                [
                    (I2, J1),
                    other.W_A,
                    (I2, 2 * I1),
                    seye(I2),
                    -seye(I2),
                    (I2, 2 * Ip1 + 2 * Ip2),
                ],
                [(Inew, J1 + J2), seye(Inew)],
            ),
            W_Ap=self.cat(
                [
                    self.W_Ap,
                    (Ip1, J2 + 2 * I1 + 2 * I2),
                    1 / 2 * seye(Ip1),
                    -1 / 2 * seye(Ip1),
                    (Ip1, 2 * Ip2),
                ],
                [
                    (Ip2, J1),
                    other.W_Ap,
                    (Ip2, 2 * Ip1 + 2 * I1 + 2 * I2),
                    1 / 2 * seye(Ip2),
                    -1 / 2 * seye(Ip2),
                ],
                [
                    -1 / 2 * (self.W_B.to_sparse_coo() + self.W_Ap.sum(0)).unsqueeze(0),
                    1
                    / 2
                    * (other.W_B.to_sparse_coo() + other.W_Ap.sum(0)).unsqueeze(0),
                    sones(2 * I1).unsqueeze(0),
                    -sones(2 * I2).unsqueeze(0),
                    sones(2 * Ip1).unsqueeze(0),
                    -sones(2 * Ip2).unsqueeze(0),
                ],
            ),
            W_b=self.cat(
                [
                    1 / 2 * (self.W_B - self.W_Ap.sum(0).to_dense()),
                    1 / 2 * (other.W_B - other.W_Ap.sum(0).to_dense()),
                    1 / 2 * self.ones(2 * I1 + 2 * I2),
                    self.zeros(
                        Ip1,
                    ),
                    self.ones(Ip1),
                    self.zeros(
                        Ip2,
                    ),
                    self.ones(Ip2),
                ],
            ),
        )

    def cat(
        self,
        *elements: List[Tensor | tuple],
        row_dims: Optional[EllipsisType | int] = None,
        column_dims: Optional[EllipsisType | int] = None,
    ) -> Tensor:
        if len(elements) == 0:
            raise ValueError("No elements provided in self.cat")

        return t.cat(
            [
                t.cat(
                    [
                        self.zeros(*j).to_sparse_coo() if isinstance(j, tuple) else j
                        for j in i
                    ],
                    dim=column_dims if column_dims is not None else -1,
                )
                for i in elements
            ],
            dim=row_dims if row_dims is not None else 0,
        )

    def to(
        self,
        device: Optional[t.device] = None,
        dtype: Optional[t.dtype] = None,
    ) -> "HCZ":
        """Torch to"""
        device = self.device if device is None else device
        dtype = self.dtype if dtype is None else dtype
        return self.clone(
            W_c=self.W_C.to(device=device, dtype=dtype),
            W_G=self.W_G.to(device=device, dtype=dtype),
            W_Gp=self.W_Gp.to(device=device, dtype=dtype),
            W_A=self.W_A.to(device=device, dtype=dtype),
            W_Ap=self.W_Ap.to(device=device, dtype=dtype),
            W_b=self.W_B.to(device=self.device, dtype=self.dtype),
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
