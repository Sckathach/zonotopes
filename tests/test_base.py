import numpy as np
import torch as t

from zonotope.zonotope import Zonotope


def test_zonotope_initialize():
    """Test basic Zonotope initialization"""
    center = [1.0, 2.0, 3.0]
    infinity_terms = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    special_terms = [[0.01, 0.02], [0.03, 0.04], [0.05, 0.06]]

    # Zonotope with only center
    z = Zonotope.from_values(center)
    assert t.allclose(z.W_C, t.tensor(center))
    assert z.W_Ei.shape[-1] == 0
    assert z.W_Es.shape[-1] == 0

    # Zonotope with center and infinity terms
    z = Zonotope.from_values(center, infinity_terms=infinity_terms)
    assert t.allclose(z.W_C, t.tensor(center))
    assert t.allclose(z.W_Ei, t.tensor(infinity_terms))

    # Complete zonotope
    z = Zonotope.from_values(
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

    z = Zonotope.from_values(center, infinity_terms, special_terms, p=2)

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
