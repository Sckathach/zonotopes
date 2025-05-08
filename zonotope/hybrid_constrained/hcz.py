import textwrap
from functools import partial
from typing import Any, Literal, Optional, Tuple, Union

import einops
import torch as t
from jaxtyping import Float
from torch import Tensor

from zonotope.utils import get_einops_pattern_for_error_terms


class HCZ:
    def __init__(
        self,
        center: Float[Tensor, "..."],
        continuous_generators: Optional[Float[Tensor, "... Ng"]] = None,
        binary_generators: Optional[Float[Tensor, "... Nb"]] = None,
        continuous_constraints: Optional[Float[Tensor, "Nc Ng"]] = None,
        binary_constraints: Optional[Float[Tensor, "Nc Nb"]] = None,
        constraints_biases: Optional[Float[Tensor, "Nc"]] = None,
        clone: bool = True,
    ) -> None:
        self.W_C: Float[Tensor, "..."] = center.clone() if clone else center

        if (
            continuous_generators is None or continuous_generators.shape[-1] == 0
        ):  # second condition to reset zeros'shape
            self.W_Gc: Float[Tensor, "... Ng"] = self.zeros(*self.shape, 0)
        else:
            self.W_Gc = (
                continuous_generators.clone() if clone else continuous_generators
            )

        if binary_generators is None or binary_generators.shape[-1] == 0:
            self.W_Gb: Float[Tensor, "... Nb"] = self.zeros(*self.shape, 0)
        else:
            self.W_Gb = binary_generators.clone() if clone else binary_generators

        if constraints_biases is None or constraints_biases.shape[-1] == 0:
            self.W_B: Float[Tensor, "Nc"] = self.zeros(0)
        else:
            self.W_B = constraints_biases.clone() if clone else constraints_biases

        if continuous_constraints is None or continuous_constraints.shape[-1] == 0:
            self.W_Ac: Float[Tensor, "Nc Ng"] = self.zeros(self.Nc, self.Ng)
        else:
            self.W_Ac = (
                continuous_constraints.clone() if clone else continuous_constraints
            )

        if binary_constraints is None or binary_constraints.shape[-1] == 0:
            self.W_Ab: Float[Tensor, "Nc Nb"] = self.zeros(self.Nc, self.Nb)
        else:
            self.W_Ab = binary_constraints.clone() if clone else binary_constraints

    @property
    def Nc(self) -> int:
        return self.W_B.shape[-1]

    @property
    def Nb(self) -> int:
        return self.W_Gb.shape[-1]

    @property
    def Ng(self) -> int:
        return self.W_Gc.shape[-1]

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

    def zeros(self, *shape, **kwargs) -> Float[Tensor, "..."]:
        kwargs = {"device": self.device, "dtype": self.dtype} | kwargs
        return t.zeros(*shape, **kwargs)  # type: ignore

    def display_shapes(self) -> None:
        print(
            textwrap.dedent(f"""
                c: {self.W_C.shape}
                Gc: {self.W_Gc.shape}
                Gb: {self.W_Gb.shape}
                b: {self.W_B.shape}
                Ac: {self.W_Ac.shape}
                Ab: {self.W_Ab.shape}
            """)
        )

    def display_weights(self) -> str:
        return textwrap.dedent(f"""
            c: {self.W_C}
            Gc: {self.W_Gc}
            Gb: {self.W_Gb}
            b: {self.W_B}
            Ac: {self.W_Ac}
            Ab: {self.W_Ab}
        """)

    @classmethod
    def from_values(
        cls,
        center: Any,
        continuous_generators: Any = None,
        binary_generators: Any = None,
        continuous_constraints: Any = None,
        binary_constraints: Any = None,
        constraints_biases: Any = None,
        dtype: t.dtype = t.float32,
        device: Optional[t.device] = None,
    ) -> "HCZ":
        if device is None:
            device = t.device("cuda")
        as_tensor = partial(t.as_tensor, dtype=dtype, device=device)

        center_ = as_tensor(center)

        continuous_generators_ = None
        if continuous_generators is not None:
            continuous_generators_ = as_tensor(continuous_generators)

        binary_generators_ = None
        if binary_generators is not None:
            binary_generators_ = as_tensor(binary_generators)

        continuous_constraints_ = None
        if continuous_constraints is not None:
            continuous_constraints_ = as_tensor(continuous_constraints)

        binary_constraints_ = None
        if binary_constraints is not None:
            binary_constraints_ = as_tensor(binary_constraints)

        constraints_biases_ = None
        if constraints_biases is not None:
            constraints_biases_ = as_tensor(constraints_biases)

        return cls(
            center=center_,
            continuous_generators=continuous_generators_,
            binary_generators=binary_generators_,
            continuous_constraints=continuous_constraints_,
            binary_constraints=binary_constraints_,
            constraints_biases=constraints_biases_,
        )

    @classmethod
    def from_bounds(cls, lower: Any, upper: Any) -> "HCZ":
        center = (lower + upper) / 2
        radius = (upper - lower) / 2
        return cls.from_values(
            center=center,
            continuous_generators=radius.unsqueeze(-1),
        )

    def clone(
        self,
        center: Optional[Float[Tensor, "..."]] = None,
        continuous_generators: Optional[Float[Tensor, "... Ng"]] = None,
        binary_generators: Optional[Float[Tensor, "... Nb"]] = None,
        continuous_constraints: Optional[Float[Tensor, "Nc Ng"]] = None,
        binary_constraints: Optional[Float[Tensor, "Nc Nb"]] = None,
        constraints_biases: Optional[Float[Tensor, "Nc"]] = None,
    ) -> "HCZ":
        return HCZ(
            center=self.W_C if center is None else center,
            continuous_generators=self.W_Gc
            if continuous_generators is None
            else continuous_generators,
            binary_generators=self.W_Gb
            if binary_generators is not None
            else binary_generators,
            continuous_constraints=self.W_Ac
            if continuous_constraints is None
            else continuous_constraints,
            binary_constraints=self.W_Ab
            if binary_constraints is None
            else binary_constraints,
            constraints_biases=self.W_B
            if constraints_biases is None
            else constraints_biases,
        )

    def concretize(self, **kwargs) -> Tuple[Float[Tensor, "N"], Float[Tensor, "N"]]:
        if self.Nc == 0:
            return self.dual(self.zeros(self.N, self.Nc), "upper"), -self.dual(
                self.zeros(self.N, self.Nc)
            )
        lambda_lower = self.optimize_lambda("lower", **kwargs)
        lambda_upper = self.optimize_lambda("upper", **kwargs)
        return self.dual(lambda_lower, "lower"), -self.dual(lambda_upper, "upper")

    def dual(
        self, lmda: Float[Tensor, "N Nc"], bound: Literal["upper", "lower"] = "lower"
    ) -> Float[Tensor, "N"]:
        if bound == "lower":
            return (
                self.W_C
                + einops.einsum(lmda, self.W_B, "N Nc, Nc -> N")
                - t.linalg.norm(
                    self.W_Gc - einops.einsum(lmda, self.W_Ac, "N Nc, Nc Ng -> N Ng"),
                    ord=1,
                    dim=-1,
                )
                - t.linalg.norm(
                    self.W_Gb - einops.einsum(lmda, self.W_Ab, "N Nc, Nc Nb -> N Nb"),
                    ord=1,
                    dim=-1,
                )
            )
        else:
            return (
                -self.W_C
                + einops.einsum(lmda, self.W_B, "N Nc, Nc -> N")
                - t.linalg.norm(
                    -self.W_Gc - einops.einsum(lmda, self.W_Ac, "N Nc, Nc Ng -> N Ng"),
                    ord=1,
                    dim=-1,
                )
                - t.linalg.norm(
                    -self.W_Gb - einops.einsum(lmda, self.W_Ab, "N Nc, Nc Nb -> N Nb"),
                    ord=1,
                    dim=-1,
                )
            )

    def optimize_lambda(
        self,
        bound: Literal["upper", "lower"] = "lower",
        num_iterations: int = 1000,
        learning_rate: float = 1e-3,
    ) -> Float[Tensor, "N Nc"]:
        lmda = (
            t.randn(
                self.N,
                self.Nc,
                device=self.device,
                dtype=self.dtype,
            )
            * 1e-4
        ).requires_grad_(True)

        optimizer = t.optim.Adam([lmda], lr=learning_rate)

        best_lmda = lmda.clone().detach()
        best_value = float("-inf")

        for iteration in range(num_iterations):
            optimizer.zero_grad()

            current_bound = self.dual(lmda, bound=bound)
            loss = -current_bound.sum()

            if t.isnan(loss).any():
                print(
                    textwrap.dedent(f"""
                    NaN detected at iteration {iteration}""
                    lmda stats: min={lmda.min().item()}, max={lmda.max().item()}
                    concretize stats: {self.dual(lmda)}
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

            if iteration % 100 == 0:
                print(f"Iteration {iteration}, Concretize Sum: {-loss.item()}")

        return best_lmda

    def add(self, other: Union["HCZ", float, int, Tensor]) -> "HCZ":
        if isinstance(other, HCZ):
            new_ac_top = t.cat([self.W_Ac, self.zeros(self.Nc, other.Ng)])
            new_ac_bottom = t.cat(
                [
                    self.zeros(other.Nc, self.Ng),
                    other.W_Ac,
                ]
            )
            new_ac = t.cat([new_ac_top, new_ac_bottom], dim=0)

            new_ab_top = t.cat(
                [
                    self.W_Ab,
                    self.zeros(self.Nc, other.Nb),
                ]
            )
            new_ab_bottom = t.cat(
                [
                    self.zeros(other.Nc, self.Nb),
                    other.W_Ab,
                ]
            )
            new_ab = t.cat([new_ab_top, new_ab_bottom], dim=0)

            return self.clone(
                center=self.W_C + other.W_C,
                continuous_generators=t.cat([self.W_Gc, other.W_Gc], dim=-1),
                binary_generators=t.cat([self.W_Gb, other.W_Gb], dim=-1),
                continuous_constraints=new_ac,
                binary_constraints=new_ab,
                constraints_biases=t.cat([self.W_B, other.W_B], dim=-1),
            )

        return self.clone(center=self.W_C + other)

    def mul(self, other: Union[float, int, Tensor]) -> "HCZ":
        if isinstance(other, Tensor):
            return self.clone(
                center=self.W_C * other,
                continuous_generators=self.W_Gc * other.unsqueeze(-1),
                binary_generators=self.W_Gb * other.unsqueeze(-1),
            )
        else:
            return self.clone(
                center=self.W_C * other,
                continuous_generators=self.W_Gc * other,
                binary_generators=self.W_Gb * other,
            )

    def sub(self, other: Union["HCZ", float, int, Tensor]) -> "HCZ":
        return self + (-1 * other)

    def rsub(self, other: Union["HCZ", float, int, Tensor]) -> "HCZ":
        return other + (-1 * self)

    def div(self, other: Union[float, int, Tensor]) -> "HCZ":
        return self * (1 / other)

    def intersect(self, other: "HCZ") -> "HCZ":
        new_gc = t.cat([self.W_Gc, self.zeros(*self.shape, other.Ng)], dim=-1)
        new_gb = t.cat([self.W_Gb, self.zeros(*self.shape, other.Nb)], dim=-1)

        new_ac_top = t.cat([self.W_Ac, self.zeros(self.Nc, other.Ng)], dim=-1)
        new_ac_mid = t.cat([self.zeros(other.Nc, self.Ng), other.W_Ac], dim=-1)
        new_ac_bottom = t.cat([self.W_Gc, -other.W_Gc], dim=-1)
        new_ac = t.cat([new_ac_top, new_ac_mid, new_ac_bottom], dim=0)

        new_ab_top = t.cat([self.W_Ab, self.zeros(self.Nc, other.Nb)], dim=-1)
        new_ab_mid = t.cat([self.zeros(other.Nc, self.Nb), other.W_Ab], dim=-1)
        new_ab_bottom = t.cat([self.W_Gb, -other.W_Gb], dim=-1)
        new_ab = t.cat([new_ab_top, new_ab_mid, new_ab_bottom], dim=0)

        new_b = t.cat([self.W_B, other.W_B, other.W_C - self.W_C])

        return self.clone(
            center=self.W_C,
            continuous_generators=new_gc,
            binary_generators=new_gb,
            continuous_constraints=new_ac,
            binary_constraints=new_ab,
            constraints_biases=new_b,
        )

    def rearrange(self, pattern: str, **kwargs) -> "HCZ":
        """Einops rearrange"""
        error_pattern = get_einops_pattern_for_error_terms(pattern)
        return self.clone(
            center=einops.rearrange(self.W_C, pattern, **kwargs),
            continuous_generators=einops.rearrange(self.W_Gc, error_pattern, **kwargs),
            binary_generators=einops.rearrange(self.W_Gb, error_pattern, **kwargs),
        )

    def repeat(self, pattern: str, **kwargs) -> "HCZ":
        """Einops repeat"""
        error_pattern = get_einops_pattern_for_error_terms(pattern)
        return self.clone(
            center=einops.repeat(self.W_C, pattern, **kwargs),
            continuous_generators=einops.repeat(self.W_Gc, error_pattern, **kwargs),
            binary_generators=einops.repeat(self.W_Gb, error_pattern, **kwargs),
        )

    def einsum(self, other: Tensor, pattern: str, **kwargs) -> "HCZ":
        """Einops einsum"""
        error_pattern = get_einops_pattern_for_error_terms(pattern)
        return self.clone(
            center=einops.einsum(self.W_C, other, pattern, **kwargs),
            continuous_generators=einops.einsum(
                self.W_Gc, other, error_pattern, **kwargs
            ),
            binary_generators=einops.einsum(self.W_Gb, other, error_pattern, **kwargs),
        )

    def to(
        self, device: Optional[t.device] = None, dtype: Optional[t.dtype] = None
    ) -> "HCZ":
        """Torch to"""
        device = self.device if device is None else device
        dtype = self.dtype if dtype is None else dtype
        return self.clone(
            center=self.W_C.to(device=device, dtype=dtype),
            continuous_generators=self.W_Gc.to(device=device, dtype=dtype),
            binary_generators=self.W_Gb.to(device=device, dtype=dtype),
            continuous_constraints=self.W_Ac.to(device=device, dtype=dtype),
            binary_constraints=self.W_Ab.to(device=device, dtype=dtype),
            constraints_biases=self.W_B.to(device=self.device, dtype=self.dtype),
        )

    def contiguous(self) -> None:
        """Torch contiguous"""
        self.W_C.contiguous()
        self.W_Gc.contiguous()
        self.W_Gb.contiguous()
        self.W_Ac.contiguous()
        self.W_Ab.contiguous()
        self.W_B.contiguous()

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
