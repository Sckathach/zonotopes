# import torch as t
# from einops import einsum

# from tests.utils import empirical_soundness
# from zonotope.classical.functional import (
#     dot_product,
#     exp,
#     reciprocal,
#     relu,
#     softmax,
#     tanh,
# )
# from zonotope.classical.z import Zonotope


# def test_relu_transformer_all_positive():
#     """Test ReLU transformer when all bounds are positive"""
#     # Create a zonotope with all positive bounds
#     z = Zonotope.from_values([2.0, 3.0, 4.0], W_G=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

#     lower, upper = z.concretize()
#     assert t.all(lower > 0)  # Verify all bounds are positive

#     result = relu(z)

#     # For positive bounds, ReLU should be identity
#     assert t.allclose(result.W_C, z.W_C)
#     assert t.allclose(result.W_G, z.W_G)
#     assert t.allclose(result.W_Es, z.W_Es)


# def test_relu_transformer_all_negative():
#     """Test ReLU transformer when all bounds are negative"""
#     # Create a zonotope with all negative bounds
#     z = Zonotope.from_values(
#         [-2.0, -3.0, -4.0], W_G=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
#     )

#     lower, upper = z.concretize()
#     assert t.all(upper < 0)  # Verify all bounds are negative

#     result = relu(z)

#     # For negative bounds, ReLU should output zeros
#     assert t.allclose(result.W_C, t.zeros_like(z.W_C))
#     assert t.allclose(result.W_G, t.zeros_like(z.W_G))
#     assert t.allclose(result.W_Es, t.zeros_like(z.W_Es))


# def test_relu_transformer_crossing_zero():
#     """Test ReLU transformer when bounds cross zero"""
#     # Create a zonotope with bounds crossing zero
#     z = Zonotope.from_values([-1.0, 0.0, 1.0], W_G=[[2.0, 0.0], [2.0, 0.0], [0.5, 0.0]])

#     lower, upper = z.concretize()

#     assert t.any((lower < 0) & (upper > 0))

#     result = relu(z)

#     # Check that a new error term has been added for crossing cases
#     assert result.Ei > z.Ei

#     empirical_soundness(z, result, t.relu)


# def test_tanh_transformer():
#     """Test tanh transformer"""
#     z = Zonotope.from_values([-2.0, 0.0, 2.0], W_G=[[0.5, 0.0], [1.0, 0.0], [0.5, 0.0]])

#     result = tanh(z)
#     assert result.Ei == z.Ei + 1, "New term should be added"

#     # Test that bounds are reasonable (tanh maps to [-1, 1])
#     lower, upper = result.concretize()
#     assert t.all(lower >= -1 - 1e-5)
#     assert t.all(upper <= 1 + 1e-5)

#     empirical_soundness(z, result, t.tanh)


# def test_exp_transformer():
#     """Test exponential transformer"""
#     z = Zonotope.from_values([-1.0, 0.0, 1.0], W_G=[[0.5, 0.0], [0.5, 0.0], [0.5, 0.0]])

#     result = exp(z)
#     assert result.Ei == z.Ei + 1, "New term should be added"

#     lower, _ = result.concretize()
#     assert t.all(lower >= 0), "The resulting lower bound should be positive"

#     empirical_soundness(z, result, t.exp)


# def test_reciprocal_transformer():
#     """Test reciprocal transformer"""
#     # Create test zonotope with positive bounds
#     z = Zonotope.from_values([1.0, 2.0, 3.0], W_G=[[0.2, 0.0], [0.5, 0.0], [0.5, 0.0]])

#     lower, _ = z.concretize()
#     result = reciprocal(z)

#     assert result.Ei == z.Ei + 1, "New term should be added"

#     lower, _ = result.concretize()
#     assert t.all(lower >= 0), "The resulting lower bound should be positive"

#     empirical_soundness(z, result, lambda x: 1 / x)


# def test_dot_product():
#     z = Zonotope.from_values([1.0, 2.0, 3.0], W_G=[[0.2, 0.0], [0.5, 0.0], [0.5, 0.0]])

#     result = dot_product(z, z, pattern="a, a -> ")

#     assert result.Ei == z.Ei + 1, "New term should be added"

#     empirical_soundness(
#         z, result, lambda x, y: einsum(x, y, "batch n, batch n -> batch"), b=z
#     )


# def test_softmax():
#     z = Zonotope.from_values([1.0, 2.0, 3.0], W_G=[[0.2, 0.0], [0.5, 0.0], [0.0, 1.0]])

#     result = softmax(z)

#     empirical_soundness(z, result, lambda x: x.softmax(-1))
