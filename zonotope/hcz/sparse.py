import math
from typing import Any, Literal, Optional, Self

import torch as t
from jaxtyping import Float
from torch import Tensor
from torch.linalg import norm

from zonotope import DEFAULT_DEVICE, DEFAULT_DTYPE
from zonotope.hcz.base import HCZBase, HCZConfig
from zonotope.utils import parse_einops_pattern


class HCZSparse(HCZBase):
    def __init__(
        self,
        W_C: Float[Tensor, "N"],
        W_G: Optional[Float[Tensor, "N I"]] = None,
        W_Gp: Optional[Float[Tensor, "N Ip"]] = None,
        W_A: Optional[Float[Tensor, "J I"]] = None,
        W_Ap: Optional[Float[Tensor, "J Ip"]] = None,
        W_B: Optional[Float[Tensor, "J"]] = None,
        virtual_shape: Optional[t.Size] = None,
        config: Optional[HCZConfig] = None,
        clone: bool = True,
        **kwargs,
    ) -> None:
        self.virtual_shape = virtual_shape if virtual_shape is not None else W_C.shape
        w_c = W_C.reshape(-1)
        super().__init__(w_c, W_G, W_Gp, W_A, W_Ap, W_B, config, clone, **kwargs)
        self.to_sparse_()

    def check_integrity(self) -> None:
        super().check_integrity()
        assert not self.W_C.is_sparse
        assert not self.W_B.is_sparse
        assert self.W_G.is_sparse
        assert self.W_Gp.is_sparse
        assert self.W_A.is_sparse
        assert self.W_Ap.is_sparse
        assert self.W_C.shape == t.Size([self.N])
        assert self.W_G.shape == t.Size([self.N, self.I])
        assert self.W_Gp.shape == t.Size([self.N, self.Ip])
        assert self.W_A.shape == t.Size([self.J, self.I])
        assert self.W_Ap.shape == t.Size([self.J, self.Ip])

    def to_sparse_(self) -> None:
        self.W_G = self.W_G.to_sparse_coo()
        self.W_Gp = self.W_Gp.to_sparse_coo()
        self.W_A = self.W_A.to_sparse_coo()
        self.W_Ap = self.W_Ap.to_sparse_coo()

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

        w_c = as_tensor(W_C).reshape(-1)
        shape = w_c.shape

        result = super().from_values(
            W_C=w_c,
            W_G=W_G,
            W_Gp=W_Gp,
            W_A=W_A,
            W_Ap=W_Ap,
            W_B=W_B,
            dtype=dtype,
            device=device,
            config=config,
            virtual_shape=shape,
            **kwargs,
        )
        result.to_sparse_()
        return result

    def dual(
        self, lmda: Float[Tensor, "N J"], bound: Literal["upper", "lower"] = "lower"
    ) -> Float[Tensor, "N"]:
        coeff = -1 if bound == "upper" else 1
        return (
            -norm(
                t.sparse.mm(self.W_A.T, lmda.T).T - coeff * self.W_G,
                ord=1,
                dim=-1,
            )
            - norm(
                t.sparse.mm(self.W_Ap.T, lmda.T).T - coeff * self.W_Gp,
                ord=1,
                dim=-1,
            )
            + coeff * self.W_C
            + lmda @ self.W_B
        )

    def intersect(
        self,
        other: Self,
        r: Optional[Float[Tensor, "N M"]] = None,
        check_emptiness_before: bool = True,
        check_emptiness_after: bool = True,
        **kwargs,
    ) -> Self:
        if check_emptiness_before and (
            self.is_empty(**kwargs) or other.is_empty(**kwargs) == 0
        ):
            return self.empty_from_self()
        if r is not None:
            rg = t.sparse.mm(self.W_G.T, r.to_sparse_coo().T).T
            rgp = t.sparse.mm(self.W_Gp.T, r.to_sparse_coo().T).T
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
            W_B=self.cat([self.W_B, other.W_B, other.W_C - rc]),
        )
        if check_emptiness_after and result.is_empty(**kwargs):
            return self.empty_from_self()

        return result

    def cartesian_product(
        self,
        other: Self,
        check_emptiness: bool = True,
        new_virtual_shape: Optional[t.Size] = None,
        **kwargs,
    ) -> Self:
        new_virtual_shape = (
            t.Size([self.N + other.N])
            if new_virtual_shape is None
            else new_virtual_shape
        )
        result = super().cartesian_product(other, check_emptiness, **kwargs)
        result.virtual_shape = new_virtual_shape
        return result

    def union(self, other: Self, check_emptiness: bool = True, **kwargs) -> Self:
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
            W_C=1
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
            W_B=self.cat(
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

    def mm(self, other: Float[Tensor, "A B"]) -> Self:
        return self.einsum(other)

    def einsum(
        self, other: Float[Tensor, "A B"], virtual_pattern: Optional[str] = None
    ) -> Self:
        if virtual_pattern is not None:
            patterns = parse_einops_pattern(
                virtual_pattern, shape_a=self.virtual_shape, shape_b=other.shape
            )

            final_shape = patterns["out_shape"]
            dim = patterns["removed_dim"]
        else:
            final_shape = [other.shape[-1]]
            dim = -1

        initial_shape = list(self.virtual_shape)
        ma, mb = other.shape

        initial_shape_left = initial_shape.copy()
        dim_value = initial_shape_left.pop(dim)
        initial_shape_left += [dim_value]
        permute_dim = list(range(len(initial_shape) + 1))
        permute_dim.pop(dim)
        permute_dim = permute_dim[:-1] + [dim] + permute_dim[-1:]

        other_expanded = self.zeros(math.prod(initial_shape), math.prod(final_shape))
        for i in range(math.prod(initial_shape_left[:-1])):
            other_expanded[i * ma : (i + 1) * ma, i * mb : (i + 1) * mb] = other

        other_expanded_permuted = (
            other_expanded.reshape(*initial_shape_left, math.prod(final_shape))
            .permute(*permute_dim)
            .reshape(math.prod(initial_shape), math.prod(final_shape))
        )  # N M

        return self.clone(
            W_C=self.W_C @ other_expanded_permuted,
            W_G=t.sparse.mm(self.W_G, other_expanded_permuted),
            W_Gp=t.sparse.mm(self.W_Gp, other_expanded_permuted),
        )
