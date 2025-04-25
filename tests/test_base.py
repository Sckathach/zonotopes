"""
! WARNING: LLM generated !

(but tested)
"""

import pytest
import torch as t
from einops import einsum

from zonotope.zonotope import Zonotope, dual_norm


@pytest.fixture
def device():
    """Fixture for device to run tests on."""
    return t.device("cuda" if t.cuda.is_available() else "cpu")


@pytest.fixture
def simple_zonotope(device):
    """Fixture for a simple 2D zonotope with both types of error terms."""
    center = t.tensor([1.0, 2.0], device=device)
    infinity_terms = t.tensor([[0.1, 0.2], [0.3, 0.4]], device=device)
    special_terms = t.tensor([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]], device=device)

    return Zonotope(
        center=center,
        infinity_terms=infinity_terms,
        special_terms=special_terms,
        special_norm=2,
    )


@pytest.fixture
def batch_zonotope(device):
    """Fixture for a batched zonotope (3D) with both types of error terms."""
    center = t.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device=device)
    infinity_terms = t.tensor(
        [[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]], [[0.9, 1.0], [1.1, 1.2]]],
        device=device,
    )
    special_terms = t.tensor(
        [
            [[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]],
            [[0.07, 0.08, 0.09], [0.10, 0.11, 0.12]],
            [[0.13, 0.14, 0.15], [0.16, 0.17, 0.18]],
        ],
        device=device,
    )

    return Zonotope(
        center=center,
        infinity_terms=infinity_terms,
        special_terms=special_terms,
        special_norm=2,
    )


@pytest.fixture
def zonotope_no_inf(device):
    """Fixture for a zonotope with only special error terms."""
    center = t.tensor([1.0, 2.0], device=device)
    special_terms = t.tensor([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]], device=device)

    return Zonotope(center=center, special_terms=special_terms, special_norm=2)


@pytest.fixture
def zonotope_no_special(device):
    """Fixture for a zonotope with only infinity error terms."""
    center = t.tensor([1.0, 2.0], device=device)
    infinity_terms = t.tensor([[0.1, 0.2], [0.3, 0.4]], device=device)

    return Zonotope(center=center, infinity_terms=infinity_terms, special_norm=2)


class TestZonotope:
    """Tests for the Zonotope class."""

    def test_initialization(self, device):
        """Test basic initialization with different combinations of parameters."""
        # Test with all parameters
        center = t.tensor([1.0, 2.0], device=device)
        infinity_terms = t.tensor([[0.1, 0.2], [0.3, 0.4]], device=device)
        special_terms = t.tensor([[0.01, 0.02], [0.03, 0.04]], device=device)

        z = Zonotope(
            center=center,
            infinity_terms=infinity_terms,
            special_terms=special_terms,
            special_norm=2,
        )

        assert t.allclose(z.W_C, center)
        assert t.allclose(z.W_Ei, infinity_terms)
        assert t.allclose(z.W_Es, special_terms)
        assert z.p == 2
        assert z.q == 2  # Dual of 2-norm is 2

        # Test with only center
        z = Zonotope(center=center)
        assert t.allclose(z.W_C, center)
        assert z.Ei == 0
        assert z.Es == 0

        # Test with non-default p-norm
        z = Zonotope(center=center, special_norm=3)
        assert z.p == 3
        assert z.q == 1.5  # Dual of 3-norm is 3/2

        # Test without cloning
        z = Zonotope(center=center, clone=False)
        assert z.W_C is center  # Same object, not a clone

    def test_properties(self, simple_zonotope):
        """Test all property accessors."""
        z = simple_zonotope

        # Basic properties
        assert z.Ei + z.Es == z.E
        assert z.Ei == 2
        assert z.Es == 3
        assert z.N == 2
        assert z.shape == t.Size([2])
        assert z.dtype == t.float32

        # Ensure device is correct
        assert z.device == z.W_C.device

    def test_from_values(self, device):
        """Test the from_values class method."""
        # Test with list inputs
        z = Zonotope.from_values(
            center=[1.0, 2.0],
            infinity_terms=[[0.1, 0.2], [0.3, 0.4]],
            special_terms=[[0.01, 0.02], [0.03, 0.04]],
            special_norm=2,
        )

        assert z.W_C.device.type == "cpu"
        assert z.W_C.shape == t.Size([2])
        assert z.W_Ei.shape == t.Size([2, 2])
        assert z.W_Es.shape == t.Size([2, 2])

        # Test with numpy arrays if available
        try:
            import numpy as np

            center_np = np.array([1.0, 2.0])
            z = Zonotope.from_values(center=center_np)
            assert isinstance(z.W_C, t.Tensor)
            assert z.W_C.shape == t.Size([2])
        except ImportError:
            pass

    def test_from_bounds(self, device):
        """Test the from_bounds class method."""
        lower = t.tensor([0.0, 1.0], device=device)
        upper = t.tensor([2.0, 3.0], device=device)

        z = Zonotope.from_bounds(lower, upper, special_norm=2)

        # Check center is midpoint of bounds
        assert t.allclose(z.W_C, t.tensor([1.0, 2.0], device=device))

        # Check radius is half the difference
        expected_radius = t.tensor([[1.0], [1.0]], device=device)
        assert t.allclose(z.W_Ei, expected_radius)

        # No special terms by default
        assert z.Es == 0

    def test_concretize(self, simple_zonotope):
        """Test the concretize method."""
        z = simple_zonotope
        lower, upper = z.concretize()

        # Calculate expected bounds manually
        center = z.W_C
        inf_contribution = t.sum(t.abs(z.W_Ei), dim=1)
        special_contribution = t.linalg.norm(z.W_Es, ord=z.q, dim=1)

        expected_lower = center - inf_contribution - special_contribution
        expected_upper = center + inf_contribution + special_contribution

        assert t.allclose(lower, expected_lower)
        assert t.allclose(upper, expected_upper)

    def test_expand_infinity_error_terms(self, simple_zonotope):
        """Test expanding infinity error terms."""
        z = simple_zonotope.clone()
        original_ei = z.Ei

        # Expand to larger size
        z.expand_infinity_error_terms(5)
        assert z.Ei == 5

        # First original_ei columns should match original tensor
        assert t.allclose(z.W_Ei[:, :original_ei], simple_zonotope.W_Ei)

        # New columns should be zeros
        assert t.allclose(
            z.W_Ei[:, original_ei:], t.zeros(2, 5 - original_ei, device=z.device)
        )

        # Expanding to smaller size should have no effect
        z.expand_infinity_error_terms(3)
        assert z.Ei == 5  # Still 5, not reduced

    def test_clone(self, simple_zonotope):
        """Test cloning a zonotope."""
        z = simple_zonotope
        z_clone = z.clone()

        # Should have same values but different tensor objects
        assert t.allclose(z_clone.W_C, z.W_C)
        assert t.allclose(z_clone.W_Ei, z.W_Ei)
        assert t.allclose(z_clone.W_Es, z.W_Es)
        assert z_clone.p == z.p

        # Modifying clone should not affect original
        z_clone.W_C[0] = 99.0
        assert not t.allclose(z_clone.W_C, z.W_C)

    def test_add(self, simple_zonotope):
        """Test adding zonotopes and scalars."""
        z = simple_zonotope

        # Test adding a scalar
        result = z + 1.0
        assert t.allclose(result.W_C, z.W_C + 1.0)
        assert t.allclose(result.W_Ei, z.W_Ei)  # Error terms unchanged
        assert t.allclose(result.W_Es, z.W_Es)  # Error terms unchanged

        # Test adding another zonotope
        z2 = z.clone()
        result = z + z2
        assert t.allclose(result.W_C, z.W_C + z2.W_C)
        assert t.allclose(result.W_Ei, z.W_Ei + z2.W_Ei)
        assert t.allclose(result.W_Es, z.W_Es + z2.W_Es)

        # Test right addition with scalar
        result = 2.0 + z
        assert t.allclose(result.W_C, 2.0 + z.W_C)

    def test_add_mismatched_error_terms(self, simple_zonotope, zonotope_no_special):
        """Test adding zonotopes with different numbers of error terms."""
        with pytest.raises(AssertionError):
            # Should fail with different Es
            _ = simple_zonotope + zonotope_no_special

    def test_einsum(self, simple_zonotope, device):
        """Test multiplication with scalars and tensors."""
        z = simple_zonotope

        # Test scalar multiplication
        result = z * 2.0
        assert t.allclose(result.W_C, z.W_C * 2.0)
        assert t.allclose(result.W_Ei, z.W_Ei * 2.0)
        assert t.allclose(result.W_Es, z.W_Es * 2.0)

        # Test right multiplication
        result = 3.0 * z
        assert t.allclose(result.W_C, 3.0 * z.W_C)

        # Test tensor multiplication with einsum pattern
        weights = t.tensor([[0.5, 0.5], [0.5, 0.5]], device=device)
        pattern = "d, b d -> b"

        result = z.einsum(weights, pattern)

        # Expected: weighted sum across the dimension
        print(z.W_C.shape, z.W_Ei.shape, z.W_Es.shape)
        expected_center = einsum(z.W_C, weights, pattern)
        expected_inf = einsum(z.W_Ei, weights, "d Ei, b d -> b Ei")
        expected_special = einsum(z.W_Es, weights, "d Es, b d -> b Es")

        assert t.allclose(result.W_C, expected_center)
        assert t.allclose(result.W_Ei, expected_inf)
        assert t.allclose(result.W_Es, expected_special)

    def test_sample_point(self, simple_zonotope):
        """Test sampling points from the zonotope."""
        z = simple_zonotope
        n_samples = 10

        # Test default sampling
        samples = z.sample_point(n_samples)
        assert samples.shape == (n_samples, 2)

        # Test binary sampling
        samples = z.sample_point(n_samples, use_binary_weights=True)
        assert samples.shape == (n_samples, 2)

        # Test sampling without special terms
        samples = z.sample_point(n_samples, include_special_terms=False)
        assert samples.shape == (n_samples, 2)

        # Test sampling without infinity terms
        samples = z.sample_point(n_samples, include_infinity_terms=False)
        assert samples.shape == (n_samples, 2)

        # Just check shapes and no errors - full validation of sampling
        # distribution would need statistical tests

    def test_rearrange(self, batch_zonotope):
        """Test rearrangement operations."""
        z = batch_zonotope

        # Test simple transpose operation
        result = z.rearrange("batch dim -> dim batch")
        assert result.W_C.shape == (2, 3)  # Transposed from (3, 2)
        assert result.W_Ei.shape == (
            2,
            3,
            2,
        )  # Transposed with error dimension preserved
        assert result.W_Es.shape == (
            2,
            3,
            3,
        )  # Transposed with error dimension preserved

        # Test reshape operation
        result = z.rearrange("batch dim -> (batch dim)")
        assert result.W_C.shape == (6,)  # Flattened from (3, 2)
        assert result.W_Ei.shape == (6, 2)  # Flattened with error dimension preserved
        assert result.W_Es.shape == (6, 3)  # Flattened with error dimension preserved

    def test_repeat(self, simple_zonotope):
        """Test repetition operations."""
        z = simple_zonotope

        # Test simple repeat operation
        result = z.repeat("dim -> repeat dim", repeat=3)
        assert result.W_C.shape == (3, 2)  # Repeated from (2,)
        assert result.W_Ei.shape == (3, 2, 2)  # Repeated with error dimension preserved
        assert result.W_Es.shape == (3, 2, 3)  # Repeated with error dimension preserved

    def test_sum(self, batch_zonotope):
        """Test summation operations."""
        z = batch_zonotope

        # Test sum along batch dimension
        result = z.sum(dim=0)
        assert result.W_C.shape == (2,)  # Summed from (3, 2)
        assert result.W_Ei.shape == (2, 2)  # Summed with error dimension preserved
        assert result.W_Es.shape == (2, 3)  # Summed with error dimension preserved

        # Check values for center
        expected_center_sum = t.sum(z.W_C, dim=0)
        assert t.allclose(result.W_C, expected_center_sum)

    def test_getitem(self, batch_zonotope):
        """Test slicing operations."""
        z = batch_zonotope

        # Test single index
        result = z[0]
        assert result.W_C.shape == (2,)
        assert result.W_Ei.shape == (2, 2)
        assert result.W_Es.shape == (2, 3)
        assert t.allclose(result.W_C, z.W_C[0])

        # Test slice
        result = z[:2]
        assert result.W_C.shape == (2, 2)
        assert result.W_Ei.shape == (2, 2, 2)
        assert result.W_Es.shape == (2, 2, 3)
        assert t.allclose(result.W_C, z.W_C[:2])

        # Test multi-dimensional slice
        result = z[:2, 0]
        assert result.W_C.shape == (2,)
        assert result.W_Ei.shape == (2, 2)
        assert result.W_Es.shape == (2, 3)
        assert t.allclose(result.W_C, z.W_C[:2, 0])

        # Test with ellipsis
        result = z[..., 0]
        assert result.W_C.shape == (3,)
        assert result.W_Ei.shape == (3, 2)
        assert result.W_Es.shape == (3, 3)
        assert t.allclose(result.W_C, z.W_C[..., 0])

    def test_len(self, simple_zonotope, batch_zonotope):
        """Test length (number of elements)."""
        assert len(simple_zonotope) == 2
        assert len(batch_zonotope) == 6  # 3x2


# Test utility functions
def test_dual_norm():
    """Test the dual_norm utility function."""
    assert dual_norm(1) == float("inf")
    assert dual_norm(2) == 2
    assert dual_norm(3) == 1.5
    assert dual_norm(float("inf")) == 1
