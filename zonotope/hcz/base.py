import textwrap
from typing import Any, Callable, Literal, Optional, Self, Tuple

import torch as t
import tqdm
from jaxtyping import Float
from pydantic import BaseModel, PositiveInt
from torch import Tensor

from zonotope import DEFAULT_DEVICE, DEFAULT_DTYPE
from zonotope.classical.base import ZonotopeBase
from zonotope.classical.z import Zonotope
from zonotope.hcz import DEFAULT_LR, DEFAULT_N_STEPS


class HCZConfig(BaseModel):
    lr: float = DEFAULT_LR
    n_steps: PositiveInt = DEFAULT_N_STEPS
    verbose: bool = False


class HCZBase(ZonotopeBase):
    W_Gp: Float[Tensor, "... Ip"]
    W_A: Float[Tensor, "J I"]
    W_Ap: Float[Tensor, "J Ip"]
    W_B: Float[Tensor, "J"]
    config: HCZConfig

    def __init__(
        self,
        W_C: Float[Tensor, "..."],
        W_G: Optional[Float[Tensor, "... I"]] = None,
        W_Gp: Optional[Float[Tensor, "... I"]] = None,
        W_A: Optional[Float[Tensor, "J I"]] = None,
        W_Ap: Optional[Float[Tensor, "J Ip"]] = None,
        W_B: Optional[Float[Tensor, "J"]] = None,
        config: Optional[HCZConfig] = None,
        clone: bool = True,
        **kwargs,
    ) -> None:
        self.config = config if config is not None else HCZConfig(**kwargs)
        self.W_C: Float[Tensor, "..."] = W_C.clone() if clone else W_C

        if W_G is None or W_G.shape[-1] == 0:  # second condition to reset zeros'shape
            self.W_G: Float[Tensor, "... I"] = self.zeros(*self.shape, 0)
        else:
            self.W_G = W_G.clone() if clone else W_G

        if W_Gp is None or W_Gp.shape[-1] == 0:
            self.W_Gp: Float[Tensor, "... Ip"] = self.zeros(*self.shape, 0)
        else:
            self.W_Gp = W_Gp.clone() if clone else W_Gp

        if W_B is None or W_B.shape[-1] == 0:
            self.W_B: Float[Tensor, "J"] = self.zeros(0)
        else:
            self.W_B = W_B.clone() if clone else W_B

        if W_A is None or W_A.shape[-1] == 0:
            self.W_A: Float[Tensor, "J I"] = self.zeros(self.J, self.I)
        else:
            self.W_A = W_A.clone() if clone else W_A

        if W_Ap is None or W_Ap.shape[-1] == 0:
            self.W_Ap: Float[Tensor, "J Ip"] = self.zeros(self.J, self.Ip)
        else:
            self.W_Ap = W_Ap.clone() if clone else W_Ap

    @property
    def J(self) -> int:
        return self.W_B.shape[-1]

    @property
    def Ip(self) -> int:
        return self.W_Gp.shape[-1]

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
        W_G: Optional[Float[Tensor, "... I"]] = None,
        W_Gp: Optional[Float[Tensor, "... Ip"]] = None,
        W_A: Optional[Float[Tensor, "J I"]] = None,
        W_Ap: Optional[Float[Tensor, "J Ip"]] = None,
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
            clone=True,
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
    ) -> Self:
        lower_ = t.as_tensor(lower, dtype=dtype, device=device)
        upper_ = t.as_tensor(upper, dtype=dtype, device=device)
        shape = lower_.shape
        lower_, upper_ = lower_.reshape(-1), upper_.reshape(-1)

        center = (lower_ + upper_) / 2
        radius = (upper_ - lower_) / 2
        mask = radius != 0
        N = center.shape[0]
        I = int(mask.sum().item())  # noqa: E741
        radius_non_zeros = t.eye(I, dtype=dtype, device=device) * radius[mask]
        radius_expanded = t.zeros(I, N, dtype=dtype, device=device)
        radius_expanded[:, mask] = radius_non_zeros

        return cls.from_values(
            W_c=center.reshape(*shape),
            W_G=radius_expanded.reshape(*shape, I),
            config=config,
            **kwargs,
        )

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

    def add(self, other: Self | float | int | Tensor) -> Self:  # type: ignore
        """
        !No emptiness check!
        """
        if isinstance(other, HCZBase):
            return self.clone(
                W_C=self.W_C + other.W_C,
                W_G=self.cat([self.W_G, other.W_G]),  # N, I1 + I2
                W_Gp=self.cat([self.W_Gp, other.W_Gp]),  # N, Ip1 + Ip2
                W_A=self.cat(
                    [self.W_A, (self.J, other.I)],
                    [(other.J, self.I), other.W_A],
                ),
                W_Ap=self.cat(
                    [self.W_Ap, (self.J, other.Ip)], [(other.J, self.Ip), other.W_Ap]
                ),
                W_B=self.cat([self.W_B, other.W_B]),
            )

        if isinstance(other, Tensor):
            other = other.reshape(-1)

        return self.clone(W_C=self.W_C + other)

    def mul(self, other: float | int | Tensor) -> Self:
        """
        !No emptiness check!
        """
        if isinstance(other, Tensor):
            return self.clone(
                W_C=self.W_C * other,
                W_G=self.W_G * other.unsqueeze(-1),
                W_Gp=self.W_Gp * other.unsqueeze(-1),
            )
        else:
            return self.clone(
                W_C=self.W_C * other,
                W_G=self.W_G * other,
                W_Gp=self.W_Gp * other,
            )

    __repr__ = display_weights
    __str__ = display_weights

    def is_empty(self, **kwargs) -> bool:
        if self.N == 0:
            return True

        lower, upper = self.concretize(**kwargs)
        return bool(t.any(lower > upper).item())

    def dual(
        self, lmda: Float[Tensor, "..."], bound: Literal["upper", "lower"] = "lower"
    ):
        raise NotImplementedError

    def concretize(self, **kwargs) -> Tuple[Float[Tensor, "..."], Float[Tensor, "..."]]:
        """
        Returns: lower, upper
        """
        lower_0, upper_0 = (
            self.dual(self.zeros(self.J, *self.shape), "lower"),
            -self.dual(self.zeros(self.J, *self.shape), "upper"),
        )

        if self.J == 0:
            return lower_0, upper_0

        lambda_lower = self.optimize_lambda("lower", **kwargs)
        lambda_upper = self.optimize_lambda("upper", **kwargs)
        lower, upper = (
            self.dual(lambda_lower, "lower"),
            -self.dual(lambda_upper, "upper"),
        )

        mask_lower = lower_0 > lower
        mask_upper = upper_0 < upper
        lambda_lower[:, mask_lower] = 0
        lambda_upper[:, mask_upper] = 0

        return self.dual(lambda_lower, "lower"), -self.dual(lambda_upper, "upper")

    def optimize_lambda(
        self, bound: Literal["upper", "lower"] = "lower", **kwargs
    ) -> Float[Tensor, "... J"]:
        kwargs = self.config.model_dump() | kwargs
        lmda = (
            t.randn(
                (*self.shape, self.J),
                device=self.device,
                dtype=self.dtype,
            )
            * 1e-5
        ).requires_grad_(True)

        optimizer = t.optim.Adam([lmda], lr=kwargs["lr"])

        best_lmda = lmda.clone().detach()
        best_value = float("-inf")

        for iteration in tqdm.tqdm(range(kwargs["n_steps"])):
            optimizer.zero_grad()

            current_bound = self.dual(lmda, bound)
            loss = -current_bound.sum()

            if t.isnan(loss).any():
                print(
                    textwrap.dedent(f"""
                    NaN detected at iteration {iteration}
                    lmda stats: min={lmda.min().item()}, max={lmda.max().item()}
                    concretize stats: {self.dual(lmda, bound)}
                """)
                )
                break

            loss.backward()

            # Gradient clipping to prevent exploding gradients
            t.nn.utils.clip_grad_norm_(lmda, max_norm=1.0)

            optimizer.step()

            # Tracking
            with t.no_grad():
                current_value = current_bound.sum().item()
                if current_value > best_value:
                    best_value = current_value
                    best_lmda = lmda.clone().detach()

            if iteration % 100 == 0 and kwargs["verbose"]:
                print(f"Iteration {iteration}, Concretize Sum: {-loss.item()}")

        return best_lmda

    @classmethod
    def from_classical_transformer(
        cls,
        lower: Tensor | float,
        upper: Tensor | float,
        abs_fn: Callable[[Zonotope], Zonotope],
        eps: float = 1e-5,
    ) -> Self:
        if t.abs(t.as_tensor(lower - upper)) < eps:
            z = Zonotope.from_values(W_C=t.as_tensor([lower]))
        else:
            z = Zonotope.from_bounds(
                lower=t.as_tensor([lower]), upper=t.as_tensor([upper])
            )
        r = abs_fn(z)

        if z.I > 0 and r.I > 0:
            return cls.from_values(
                [z.W_C[0].item(), r.W_C[0].item()],
                t.tensor([[z.W_G[0, 0].item(), 0], r.W_G.tolist()[0]]).T,
            )

        return cls.from_values(
            [z.W_C[0].item(), r.W_C[0].item()],
        )

    def get_permutation_matrix(self) -> Float[Tensor, "N N"]:
        r = self.zeros(self.N, self.N)
        for i in range(self.N // 2):
            r[2 * i, i] = 1
            r[-1 - 2 * i, -i - 1] = 1
        return r

    def cartesian_product(
        self, other: Self, check_emptiness: bool = True, **kwargs
    ) -> Self:
        if check_emptiness:
            if self.is_empty(**kwargs):
                return other
            if other.is_empty(**kwargs):
                return self

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

    def apply_abstract_transformer(
        self,
        abs_transformer: Callable[[Tensor | float, Tensor | float], Self],
        **kwargs,
    ) -> Self:
        lower, upper = self.concretize(**kwargs)
        z_abs = abs_transformer(lower[0], upper[0])

        for i in range(1, self.N):
            z_abs = z_abs.cartesian_product(
                abs_transformer(lower[i], upper[i]), check_emptiness=False
            )

        z_abs.load_config_from_(self)
        perm = z_abs.get_permutation_matrix()
        r_in = t.cat([self.eye(self.N), self.zeros(self.N, self.N)], dim=-1)
        perm_z_abs = z_abs.mm(perm)
        intermediate_result = perm_z_abs.intersect(
            self,
            r_in.T,
            check_emptiness_before=False,
            check_emptiness_after=False,
            **kwargs,
        )
        r_out = t.cat([self.zeros(self.N, self.N), self.eye(self.N)], dim=-1)
        return intermediate_result.mm(r_out.T)
