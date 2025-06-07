"""
! Tests written by LLM !
(but verified (slightly))
"""

import pytest
import torch as t

from tests.utils import check_bounds
from zonotope import DEFAULT_DEVICE, DEFAULT_DTYPE
from zonotope.classical.z import Zonotope


@pytest.fixture
def z_no_noise():
    return Zonotope.from_values([0, 1, -2])


@pytest.fixture
def z_simple_noise():
    return Zonotope.from_values([0, 1, -2], [[1, 1, 0], [1, 0, 0], [0, 0, 0]])


@pytest.fixture
def z_simple_noise_2():
    return Zonotope.from_values(
        [3, 0, -0.001], [[1, -3, 0, 1], [1, 0, 0, -2], [0, 0, 0, 5]]
    )


@pytest.fixture
def z_2d():
    return Zonotope.from_values([[1, 2], [3, 4]], [[[1, 0], [0, 1]], [[0, 1], [1, 0]]])


class TestZonotope:
    def test_initialization(self, z_no_noise):
        z = Zonotope(t.tensor([0, 1, -2], device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE))
        assert t.allclose(z_no_noise.W_C, z.W_C)

    def test_initialization_with_clone_false(self):
        W_C = t.tensor([1, 2, 3], dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE)
        W_G = t.tensor(
            [[1, 0], [0, 1], [1, 1]], dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE
        )
        z = Zonotope(W_C, W_G, clone=False)

        # Modify original tensors
        W_C[0] = 999
        W_G[0, 0] = 999

        # Should affect the zonotope since clone=False
        assert z.W_C[0] == 999
        assert z.W_G[0, 0] == 999

    def test_initialization_none_W_G(self):
        z = Zonotope(t.tensor([1, 2, 3]))
        assert z.W_G.shape == (3, 0)
        assert z.I == 0

    def test_from_values(self):
        z = Zonotope.from_values([1, 2, 3], [[1, 0], [0, 1], [1, 1]])
        z.assert_equal(W_C=[1, 2, 3], W_G=[[1, 0], [0, 1], [1, 1]])

    def test_from_values_custom_device_dtype(self):
        device = t.device("cpu")
        dtype = t.float64

        z = Zonotope.from_values(
            [1, 2, 3], [[1, 0], [0, 1], [1, 1]], device=device, dtype=dtype
        )
        assert z.device == device
        assert z.dtype == dtype

    def test_from_bounds(self):
        lower = t.tensor([1, 2, 3])
        upper = t.tensor([3, 4, 5])
        z = Zonotope.from_bounds(lower, upper)

        z.assert_equal(W_C=[2, 3, 4], W_G=[[1], [1], [1]])

    def test_properties(self, z_simple_noise, z_2d):
        # Test I property
        assert z_simple_noise.I == 3
        assert z_2d.I == 2

        # Test N property
        assert z_simple_noise.N == 3
        assert z_2d.N == 4

        # Test shape property
        assert z_simple_noise.shape == t.Size([3])
        assert z_2d.shape == t.Size([2, 2])

        # Test device and dtype
        assert z_simple_noise.device == DEFAULT_DEVICE
        assert z_simple_noise.dtype == DEFAULT_DTYPE

    def test_helper_methods(self, z_simple_noise):
        kwargs = {"device": z_simple_noise.device, "dtype": z_simple_noise.dtype}
        # Test zeros
        zeros = z_simple_noise.zeros(2, 3)
        assert zeros.shape == (2, 3)
        assert zeros.device == z_simple_noise.device
        assert zeros.dtype == z_simple_noise.dtype
        assert t.allclose(zeros, t.zeros(2, 3, **kwargs))

        # Test ones
        ones = z_simple_noise.ones(2, 2)
        assert ones.shape == (2, 2)
        assert t.allclose(ones, t.ones(2, 2, **kwargs))

        # Test eye
        eye = z_simple_noise.eye(3)
        assert eye.shape == (3, 3)
        assert t.allclose(eye, t.eye(3, **kwargs))

        # Test as_tensor
        tensor = z_simple_noise.as_tensor([1, 2, 3])
        assert tensor.device == z_simple_noise.device
        assert tensor.dtype == z_simple_noise.dtype

    def test_concretization_1(self, z_simple_noise):
        check_bounds(z_simple_noise, [-2, 0, -2], [2, 2, -2])

    def test_concretization_no_noise(self, z_no_noise):
        lower, upper = z_no_noise.concretize()
        assert t.allclose(lower, z_no_noise.W_C)
        assert t.allclose(upper, z_no_noise.W_C)

    def test_expand_infinity_error_terms(self, z_simple_noise):
        original_I = z_simple_noise.I
        z_simple_noise.expand_infinity_error_terms(5)
        assert z_simple_noise.I == 5
        # Original terms should be preserved
        assert t.allclose(
            z_simple_noise.W_G[:, :original_I],
            z_simple_noise.as_tensor([[1, 1, 0], [1, 0, 0], [0, 0, 0]]),
        )
        # New terms should be zero
        assert t.allclose(
            z_simple_noise.W_G[:, original_I:], z_simple_noise.zeros(3, 2)
        )

    def test_clone(self, z_simple_noise):
        z = z_simple_noise.clone()
        z.W_C[0] = 33
        z.W_G[0] = z.as_tensor([3, 3, 3])
        z_simple_noise.assert_equal(
            W_C=[0, 1, -2], W_G=[[1, 1, 0], [1, 0, 0], [0, 0, 0]]
        )

    def test_clone_with_custom_values(self, z_simple_noise):
        new_W_C = t.tensor([10, 20, 30], dtype=DEFAULT_DTYPE)
        new_W_G = t.tensor([[1, 2], [3, 4], [5, 6]], dtype=DEFAULT_DTYPE)

        z = z_simple_noise.clone(W_C=new_W_C, W_G=new_W_G)
        assert t.allclose(z.W_C, new_W_C)
        assert t.allclose(z.W_G, new_W_G)

    def test_add_1(self, z_no_noise, z_simple_noise):
        a = z_no_noise + z_simple_noise
        z_no_noise.W_C[0] = 33
        a.assert_equal(W_C=[0, 2, -4], W_G=[[1, 1, 0], [1, 0, 0], [0, 0, 0]])

    def test_add_2(self, z_simple_noise):
        a = z_simple_noise + 33 + z_simple_noise
        z_simple_noise.W_C[0] = -33
        a.assert_equal(W_C=[33, 35, 29], W_G=[[2, 2, 0], [2, 0, 0], [0, 0, 0]])

    def test_add_3(self, z_simple_noise, z_simple_noise_2):
        a = z_simple_noise + z_simple_noise_2
        a.assert_equal(
            W_C=[3, 1, -2.001], W_G=[[2, -2, 0, 1], [2, 0, 0, -2], [0, 0, 0, 5]]
        )

    def test_add_tensor(self, z_simple_noise):
        tensor_add = z_simple_noise.as_tensor([1, 2, 3])
        result = z_simple_noise + tensor_add
        expected_W_C = z_simple_noise.W_C + tensor_add
        result.assert_equal(W_C=expected_W_C, W_G=z_simple_noise.W_G)

    def test_radd(self, z_simple_noise):
        result = 5 + z_simple_noise
        expected_W_C = z_simple_noise.W_C + 5
        result.assert_equal(W_C=expected_W_C, W_G=z_simple_noise.W_G)

    def test_sub_1(self, z_no_noise, z_simple_noise):
        a = z_no_noise - z_simple_noise
        z_no_noise.W_C[0] = 33
        a.assert_equal(W_C=[0, 0, 0], W_G=[[-1, -1, 0], [-1, 0, 0], [0, 0, 0]])

    def test_sub_scalar(self, z_simple_noise):
        result = z_simple_noise - 5
        expected_W_C = z_simple_noise.W_C - 5
        result.assert_equal(W_C=expected_W_C, W_G=z_simple_noise.W_G)

    def test_rsub(self, z_simple_noise):
        result = 5 - z_simple_noise
        expected_W_C = 5 - z_simple_noise.W_C
        expected_W_G = -z_simple_noise.W_G
        result.assert_equal(W_C=expected_W_C, W_G=expected_W_G)

    def test_mul_scalar(self, z_simple_noise):
        result = z_simple_noise * 2
        expected_W_C = z_simple_noise.W_C * 2
        expected_W_G = z_simple_noise.W_G * 2
        result.assert_equal(W_C=expected_W_C, W_G=expected_W_G)

    def test_mul_tensor(self, z_simple_noise):
        tensor_mul = z_simple_noise.as_tensor([2, 3, 4])
        result = z_simple_noise * tensor_mul
        expected_W_C = z_simple_noise.W_C * tensor_mul
        expected_W_G = z_simple_noise.W_G * tensor_mul.unsqueeze(-1)
        result.assert_equal(W_C=expected_W_C, W_G=expected_W_G)

    def test_rmul(self, z_simple_noise):
        result = 3 * z_simple_noise
        expected_W_C = z_simple_noise.W_C * 3
        expected_W_G = z_simple_noise.W_G * 3
        result.assert_equal(W_C=expected_W_C, W_G=expected_W_G)

    def test_div(self, z_simple_noise):
        result = z_simple_noise / 2
        expected_W_C = z_simple_noise.W_C / 2
        expected_W_G = z_simple_noise.W_G / 2
        result.assert_equal(W_C=expected_W_C, W_G=expected_W_G)

    def test_sample_point_no_noise(self, z_no_noise):
        samples = z_no_noise.sample_point(n_samples=10)
        assert samples.shape == (10, 3)
        # All samples should be equal to W_C since there's no noise
        for i in range(10):
            assert t.allclose(samples[i], z_no_noise.W_C)

    def test_sample_point_with_noise(self, z_simple_noise):
        samples = z_simple_noise.sample_point(n_samples=100)
        assert samples.shape == (100, 3)

        # Check that samples are within bounds
        lower, upper = z_simple_noise.concretize()
        assert t.all(samples >= lower - 1e-6)
        assert t.all(samples <= upper + 1e-6)

    def test_sample_point_binary_weights(self, z_simple_noise):
        samples = z_simple_noise.sample_point(n_samples=100, use_binary_weights=True)
        assert samples.shape == (100, 3)

        # With binary weights, samples should be at vertices of the zonotope
        lower, upper = z_simple_noise.concretize()
        assert t.all(samples >= lower - 1e-6)
        assert t.all(samples <= upper + 1e-6)

    def test_rearrange(self, z_2d):
        result = z_2d.rearrange("h w -> (h w)")
        assert result.shape == t.Size([4])
        assert result.W_G.shape == (4, 2)

    def test_repeat(self, z_simple_noise):
        result = z_simple_noise.repeat("n -> n 2")
        assert result.shape == t.Size([3, 2])
        assert result.W_G.shape == (3, 2, 3)

    def test_einsum(self, z_2d):
        matrix = z_2d.as_tensor([[1, 2], [3, 4]])
        result = z_2d.einsum(matrix, "i j, j k -> i k")
        result.assert_equal(
            W_C=[[7.0, 10.0], [15.0, 22.0]],
            W_G=[[[1.0, 3.0], [2.0, 4.0]], [[3.0, 1.0], [4.0, 2.0]]],
        )

    def test_sum(self, z_2d):
        result = z_2d.sum(dim=0)
        assert result.shape == t.Size([2])
        expected_W_C = z_2d.W_C.sum(dim=0)
        expected_W_G = z_2d.W_G.sum(dim=0)
        result.assert_equal(W_C=expected_W_C, W_G=expected_W_G)

    def test_sum_no_noise(self, z_no_noise):
        # Reshape to 2D for testing
        z_2d_no_noise = z_no_noise.rearrange("(h w) -> h w", h=1, w=3)
        result = z_2d_no_noise.sum(dim=1)
        assert result.shape == t.Size([1])

    def test_to_device_dtype(self, z_simple_noise):
        device = t.device("cpu")
        dtype = t.float64

        result = z_simple_noise.to(device=device, dtype=dtype)
        assert result.device == device
        assert result.dtype == dtype

    def test_contiguous(self, z_simple_noise):
        # This method doesn't return anything, just ensures tensors are contiguous
        z_simple_noise.contiguous()
        assert z_simple_noise.W_C.is_contiguous()
        assert z_simple_noise.W_G.is_contiguous()

    def test_getitem_single_index(self, z_simple_noise):
        result = z_simple_noise[1]
        assert result.W_C.shape == t.Size([])
        assert result.W_G.shape == t.Size([3])
        assert t.allclose(result.W_C, z_simple_noise.W_C[1])
        assert t.allclose(result.W_G, z_simple_noise.W_G[1])

    def test_getitem_slice(self, z_simple_noise):
        result = z_simple_noise[0:2]
        assert result.W_C.shape == t.Size([2])
        assert result.W_G.shape == t.Size([2, 3])
        assert t.allclose(result.W_C, z_simple_noise.W_C[0:2])
        assert t.allclose(result.W_G, z_simple_noise.W_G[0:2])

    def test_getitem_2d(self, z_2d):
        result = z_2d[0, 1]
        assert result.W_C.shape == t.Size([])
        assert result.W_G.shape == t.Size([2])

    def test_setitem(self, z_simple_noise):
        z = z_simple_noise.clone()
        z[0] = 99
        assert z.W_C[0] == 99
        assert z.W_G[0].sum() == 99 * 3  # All error terms set to 99

    def test_assert_equal_success(self, z_simple_noise):
        # Should not raise any exception
        z_simple_noise.assert_equal(
            W_C=[0, 1, -2], W_G=[[1, 1, 0], [1, 0, 0], [0, 0, 0]]
        )

    def test_assert_equal_failure(self, z_simple_noise):
        with pytest.raises(AssertionError):
            z_simple_noise.assert_equal(W_C=[1, 1, -2])

    def test_equal_same_zonotope(self, z_simple_noise):
        z_copy = z_simple_noise.clone()
        assert z_simple_noise.equal(z_copy)
        assert z_simple_noise == z_copy

    def test_equal_different_zonotope(self, z_simple_noise, z_simple_noise_2):
        assert not z_simple_noise.equal(z_simple_noise_2)
        assert z_simple_noise != z_simple_noise_2

    def test_equal_non_zonotope(self, z_simple_noise):
        assert not z_simple_noise.equal("not a zonotope")
        assert z_simple_noise != "not a zonotope"

    def test_len(self, z_simple_noise, z_2d):
        assert len(z_simple_noise) == 3
        assert len(z_2d) == 4

    def test_cat_1(self, z_simple_noise):
        # Test the inherited cat method
        tensor1 = t.ones(2, 3)
        tensor2 = t.zeros(5, 3)
        result = z_simple_noise.cat([tensor1], [tensor2])
        assert result.shape == (7, 3)

    def test_cat_2(self, z_simple_noise):
        # Test the inherited cat method
        tensor1 = t.ones(3, 2)
        tensor2 = t.zeros(3, 5)
        result = z_simple_noise.cat([tensor1, tensor2])
        assert result.shape == (3, 7)

    def test_as_sparse_tensor(self, z_simple_noise):
        sparse = z_simple_noise.as_sparse_tensor([[1, 0, 0], [0, 1, 0]])
        assert sparse.is_sparse

    def test_empty_elements_cat_error(self, z_simple_noise):
        with pytest.raises(ValueError, match="No elements provided"):
            z_simple_noise.cat()

    def test_different_I_warning(self, z_simple_noise):
        # Create zonotope with different number of error terms
        z_different_I = Zonotope.from_values([1, 2, 3], [[1, 0], [0, 1], [1, 1]])

        # This should trigger a warning and expand error terms
        result = z_simple_noise + z_different_I
        assert max(z_simple_noise.I, z_different_I.I) == result.I

    def test_zero_I_handling(self):
        # Test zonotopes with no error terms
        z1 = Zonotope.from_values([1, 2, 3])
        z2 = Zonotope.from_values([4, 5, 6])

        result = z1 + z2
        assert result.I == 0
        result.assert_equal(W_C=[5, 7, 9])
