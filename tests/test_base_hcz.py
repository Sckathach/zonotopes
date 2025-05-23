import torch as t

from zonotope.hybrid_constrained.hcz_v2 import HCZ


def test_base():
    z = HCZ.from_values([0, 1, 2], [[1, 0], [0, 1], [1, 1]])
    lower, upper = z.concretize()
    assert t.allclose(lower, z.as_tensor([-1, 0, 0]))
    assert t.allclose(upper, z.as_tensor([1, 2, 4]))
