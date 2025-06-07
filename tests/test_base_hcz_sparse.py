# import torch as t
# from torch import Tensor

# from zonotope.hybrid_constrained.hcz_v2 import HCZ


# def is_close(a: Tensor, b: Tensor, eps: float = 0.1) -> bool:
#     return bool(t.all(t.abs(a - b) < eps).item())


# def test_intersect():
#     z1 = HCZ.from_values([2, 0], [[1, 0], [0, 1]])
#     z2 = HCZ.from_values([3, 1], [[1, 0], [0, 1]])
#     z3 = z1.intersect(z2, check_emptiness=False)
#     lower, upper = z3.concretize()

#     assert is_close(lower, z1.as_tensor([2, 0]))
#     assert is_close(upper, z1.as_tensor([3, 1]))


# def test_sum():
#     z1 = HCZ.from_values([2, 0], [[1, 0], [0, 1]])
#     z2 = HCZ.from_values([3, 1], [[1, 0], [0, 1]])
#     z3 = z1 + z2

#     assert t.allclose(z3.W_C, z3.as_tensor([5, 1]))
#     assert t.allclose(z3.W_G.to_dense(), z3.as_tensor([[1, 0], [0, 1], [1, 0], [0, 1]]))
#     assert t.allclose(z3.W_Gp.to_dense(), z3.zeros(0, 2))
#     assert t.allclose(z3.W_Ap.to_dense(), z3.zeros(0, 0))
#     assert t.allclose(z3.W_A.to_dense(), z3.zeros(4, 0))
