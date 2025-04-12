import numpy as np
import torch as t
from einops import einsum

from zonotope.functional import (
    dot_product,
    exp,
    reciprocal,
    refine_softmax_bounds,
    relu,
    softmax,
    softmax_refinement,
    tanh,
)
from zonotope.utils import create_test_zonotope, empirical_soundness


def test_zonotope_initialize():
    """Test basic Zonotope initialization"""
    center = [1.0, 2.0, 3.0]
    infinity_terms = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    special_terms = [[0.01, 0.02], [0.03, 0.04], [0.05, 0.06]]

    # Zonotope with only center
    z = create_test_zonotope(center)
    assert t.allclose(z.W_C, t.tensor(center))
    assert z.W_Ei.shape[-1] == 0
    assert z.W_Es.shape[-1] == 0

    # Zonotope with center and infinity terms
    z = create_test_zonotope(center, infinity_terms=infinity_terms)
    assert t.allclose(z.W_C, t.tensor(center))
    assert t.allclose(z.W_Ei, t.tensor(infinity_terms))

    # Complete zonotope
    z = create_test_zonotope(
        center,
        infinity_terms=infinity_terms,
        special_terms=special_terms,
        p=1,
    )
    assert t.allclose(z.W_C, t.tensor(center))
    assert t.allclose(z.W_Ei, t.tensor(infinity_terms))
    assert t.allclose(z.W_Es, t.tensor(special_terms))
    assert z.p == 1
    assert z.q == float("inf")  # dual norm of L1 is Linf


def test_zonotope_concretize():
    """Test Zonotope concretization (computing bounds)"""
    # Create a zonotope with known bounds
    center = [1.0, -2.0, 3.0]
    infinity_terms = [[0.5, 0.1], [0.2, 0.3], [0.4, 0.6]]
    special_terms = [[0.2, 0.3], [0.1, 0.4], [0.5, 0.2]]

    z = create_test_zonotope(center, infinity_terms, special_terms, p=2)

    lower, upper = z.concretize()

    # Calculate expected bounds manually
    expected_lower = []
    expected_upper = []

    for i in range(3):  # 3 dimensions
        # L1 norm for infinity terms
        infinity_contribution = sum(abs(x) for x in infinity_terms[i])

        # L2 norm for special terms (since p=2, q=2)
        special_contribution = np.sqrt(sum(x**2 for x in special_terms[i]))

        expected_lower.append(center[i] - infinity_contribution - special_contribution)
        expected_upper.append(center[i] + infinity_contribution + special_contribution)

    assert t.allclose(lower, t.tensor(expected_lower).float(), rtol=1e-5)
    assert t.allclose(upper, t.tensor(expected_upper).float(), rtol=1e-5)


def test_relu_transformer_all_positive():
    """Test ReLU transformer when all bounds are positive"""
    # Create a zonotope with all positive bounds
    z = create_test_zonotope(
        [2.0, 3.0, 4.0], infinity_terms=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    )

    lower, upper = z.concretize()
    assert t.all(lower > 0)  # Verify all bounds are positive

    result = relu(z)

    # For positive bounds, ReLU should be identity
    assert t.allclose(result.W_C, z.W_C)
    assert t.allclose(result.W_Ei, z.W_Ei)
    assert t.allclose(result.W_Es, z.W_Es)


def test_relu_transformer_all_negative():
    """Test ReLU transformer when all bounds are negative"""
    # Create a zonotope with all negative bounds
    z = create_test_zonotope(
        [-2.0, -3.0, -4.0], infinity_terms=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    )

    lower, upper = z.concretize()
    assert t.all(upper < 0)  # Verify all bounds are negative

    result = relu(z)

    # For negative bounds, ReLU should output zeros
    assert t.allclose(result.W_C, t.zeros_like(z.W_C))
    assert t.allclose(result.W_Ei, t.zeros_like(z.W_Ei))
    assert t.allclose(result.W_Es, t.zeros_like(z.W_Es))


def test_relu_transformer_crossing_zero():
    """Test ReLU transformer when bounds cross zero"""
    # Create a zonotope with bounds crossing zero
    z = create_test_zonotope(
        [-1.0, 0.0, 1.0], infinity_terms=[[2.0, 0.0], [2.0, 0.0], [0.5, 0.0]]
    )

    lower, upper = z.concretize()

    assert t.any((lower < 0) & (upper > 0))

    result = relu(z)

    # Check that a new error term has been added for crossing cases
    assert result.Ei > z.Ei

    empirical_soundness(z, result, t.relu)


def test_tanh_transformer():
    """Test tanh transformer"""
    z = create_test_zonotope(
        [-2.0, 0.0, 2.0], infinity_terms=[[0.5, 0.0], [1.0, 0.0], [0.5, 0.0]]
    )

    result = tanh(z)
    assert result.Ei == z.Ei + 1, "New term should be added"

    # Test that bounds are reasonable (tanh maps to [-1, 1])
    lower, upper = result.concretize()
    assert t.all(lower >= -1 - 1e-5)
    assert t.all(upper <= 1 + 1e-5)

    empirical_soundness(z, result, t.tanh)


def test_exp_transformer():
    """Test exponential transformer"""
    z = create_test_zonotope(
        [-1.0, 0.0, 1.0], infinity_terms=[[0.5, 0.0], [0.5, 0.0], [0.5, 0.0]]
    )

    result = exp(z)
    assert result.Ei == z.Ei + 1, "New term should be added"

    lower, _ = result.concretize()
    assert t.all(lower >= 0), "The resulting lower bound should be positive"

    empirical_soundness(z, result, t.exp)


def test_reciprocal_transformer():
    """Test reciprocal transformer"""
    # Create test zonotope with positive bounds
    z = create_test_zonotope(
        [1.0, 2.0, 3.0], infinity_terms=[[0.2, 0.0], [0.5, 0.0], [0.5, 0.0]]
    )

    lower, _ = z.concretize()
    result = reciprocal(z)

    assert result.Ei == z.Ei + 1, "New term should be added"

    lower, _ = result.concretize()
    assert t.all(lower >= 0), "The resulting lower bound should be positive"

    empirical_soundness(z, result, lambda x: 1 / x)


def test_dot_product():
    z = create_test_zonotope(
        [1.0, 2.0, 3.0], infinity_terms=[[0.2, 0.0], [0.5, 0.0], [0.5, 0.0]]
    )

    result = dot_product(z, z)

    assert result.Ei == z.Ei + 1, "New term should be added"

    empirical_soundness(z, result, lambda x, y: einsum(x, y, "N, N -> N"), b=z)


def test_affine1_concrete():
    z = create_test_zonotope([1.0, 2.0, 3.0])
    y = z.affine1(t.tensor([1.0, 2.0, 3.0]), 0.5)
    assert y.N == 1
    assert y.W_C == 1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0 + 0.5


def test_affine1_inf():
    z = create_test_zonotope(
        [1.0, 2.0, 3.0], infinity_terms=[[0.2, 0.0], [0.5, 0.3], [0.5, 0.0]]
    )
    y = z.affine1(t.tensor([1.1, 2.2, 3.3]), 0.5)
    assert y.N == 1
    assert y.W_C == 1.1 * 1.0 + 2.2 * 2.0 + 3.3 * 3.0 + 0.5
    assert t.isclose(
        y.W_Ei, t.tensor([1.1 * 0.2 + 2.2 * 0.5 + 3.3 * 0.5 + 0.5, 2.2 * 0.3 + 0.5])
    ).all()


def test_affine_concrete():
    z = create_test_zonotope([1.0, 2.0, 3.0])
    y = z.affine(
        t.tensor(
            [
                [1.1, 2.2, 3.3],
                [4.0, 5.0, 6.0],
            ]
        ).T,
        t.tensor([0.5, 2.0]),
    )
    assert t.isclose(
        y.W_C,
        t.tensor(
            [
                1.1 * 1.0 + 2.2 * 2.0 + 3.3 * 3.0 + 0.5,
                4.0 * 1.0 + 5.0 * 2.0 + 6.0 * 3.0 + 2.0,
            ]
        ),
    ).all()


def test_affine_inf():
    z = create_test_zonotope(
        [1.0, 2.0, 3.0], infinity_terms=[[0.2, 0.0], [0.5, 0.3], [0.5, 0.0]]
    )
    y = z.affine(
        t.tensor(
            [
                [1.1, 2.2, 3.3],
                [4.0, 5.0, 6.0],
            ]
        ).T,
        t.tensor([0.5, 2.0]),
    )
    assert t.isclose(
        y.W_C,
        t.tensor(
            [
                1.1 * 1.0 + 2.2 * 2.0 + 3.3 * 3.0 + 0.5,
                4.0 * 1.0 + 5.0 * 2.0 + 6.0 * 3.0 + 2.0,
            ]
        ),
    ).all()
    assert t.isclose(
        y.W_Ei,
        t.tensor(
            [
                [1.1 * 0.2 + 2.2 * 0.5 + 3.3 * 0.5 + 0.5, 2.2 * 0.3 + 0.5],
                [4.0 * 0.2 + 5.0 * 0.5 + 6.0 * 0.5 + 2.0, 5.0 * 0.3 + 2.0],
            ]
        ),
    ).all()


def test_softmax():
    z = create_test_zonotope(
        [-1.0, 0.0, 1.0], infinity_terms=[[2.0, 0.0], [1.0, 0.0], [0.0, 0.5]]
    )

    result = softmax(z)

    empirical_soundness(z, result, lambda x: t.nn.functional.softmax(x, dim=0))


def test_softmax_first_refinement():
    z = create_test_zonotope(
        [-1.0, 0.0, 1.0], infinity_terms=[[2.0, 0.0], [1.0, 0.0], [0.0, 0.5]]
    )

    result = softmax_refinement(softmax(z))

    empirical_soundness(z, result, lambda x: t.nn.functional.softmax(x, dim=0))


def test_softmax_second_refinement():
    z = create_test_zonotope(
        [-1.0, 0.0, 1.0], infinity_terms=[[2.0, 0.0], [1.0, 0.0], [0.0, 0.5]]
    )

    result = refine_softmax_bounds(softmax_refinement(softmax(z)))

    empirical_soundness(z, result, lambda x: t.nn.functional.softmax(x, dim=0))
