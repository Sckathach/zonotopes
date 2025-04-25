# Usage Guide

This guide demonstrates how to use the Zonotope package for various common tasks.

## Creating Zonotopes

There are several ways to create a Zonotope:

### Direct Initialization

```python
import torch as t
from zonotope import Zonotope

# Create a zonotope with explicit tensors
center = t.tensor([1.0, 2.0])
infinity_terms = t.tensor([[0.1, 0.2], [0.3, 0.4]])
special_terms = t.tensor([[0.01, 0.02], [0.03, 0.04]])

z = Zonotope(
    center=center,
    infinity_terms=infinity_terms,
    special_terms=special_terms,
    special_norm=2
)
```

### From Values

```python
# Create from various types of inputs (lists, numpy arrays, etc.)
z = Zonotope.from_values(
    center=[1.0, 2.0],
    infinity_terms=[[0.1, 0.2], [0.3, 0.4]],
    special_terms=[[0.01, 0.02], [0.03, 0.04]],
    special_norm=2
)
```

### From Bounds

```python
# Create from lower and upper bounds
lower = t.tensor([0.0, 1.0])
upper = t.tensor([2.0, 3.0])

z = Zonotope.from_bounds(lower, upper, special_norm=2)
```

## Basic Properties

Access various properties of a zonotope:

```python
# Shape and dimensions
print(f"Shape: {z.shape}")
print(f"Number of variables: {z.N}")
print(f"Number of infinity error terms: {z.Ei}")
print(f"Number of special error terms: {z.Es}")
print(f"Total error terms: {z.E}")

# Device and dtype
print(f"Device: {z.device}")
print(f"Data Type: {z.dtype}")

# Norms
print(f"Special norm (p): {z.p}")
print(f"Dual norm (q): {z.q}")
```

## Computing Concrete Bounds

To compute the concrete lower and upper bounds of a zonotope:

```python
lower, upper = z.concretize()
print(f"Lower bounds: {lower}")
print(f"Upper bounds: {upper}")
```

## Arithmetic Operations

Zonotopes support various arithmetic operations:

### Addition

```python
# Adding two zonotopes
z1 = Zonotope.from_values(center=[1.0, 2.0], infinity_terms=[[0.1], [0.2]])
z2 = Zonotope.from_values(center=[3.0, 4.0], infinity_terms=[[0.3], [0.4]])
z_sum = z1 + z2
print(f"Sum center: {z_sum.W_C}")

# Adding a scalar
z_shifted = z1 + 5.0
print(f"Shifted center: {z_shifted.W_C}")
```

### Multiplication

```python
# Scalar multiplication
z_scaled = z1 * 2.0
print(f"Scaled center: {z_scaled.W_C}")

# Right multiplication
z_scaled = 3.0 * z1
print(f"Scaled center: {z_scaled.W_C}")

# Tensor multiplication with einsum pattern
weights = t.tensor([[0.5, 0.5], [0.5, 0.5]])
pattern = "d, b d -> b"
z_weighted = z1.mul(weights, pattern)
print(f"Weighted center: {z_weighted.W_C}")
```

## Sampling Points

Sample points from within the zonotope:

```python
# Default sampling
samples = z.sample_point(n_samples=10)
print(f"Sampled points:\n{samples}")

# Binary sampling (corners of the zonotope)
corner_samples = z.sample_point(n_samples=10, use_binary_weights=True)
print(f"Corner samples:\n{corner_samples}")

# Sampling with only special or infinity terms
special_samples = z.sample_point(n_samples=5, include_infinity_terms=False)
infinity_samples = z.sample_point(n_samples=5, include_special_terms=False)
```

## Tensor Operations

The zonotope package supports einops-style tensor operations:

### Rearrangement

```python
# Create a batched zonotope
batch_z = Zonotope.from_values(
    center=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
    infinity_terms=[[[0.1, 0.2], [0.3, 0.4]], 
                   [[0.5, 0.6], [0.7, 0.8]], 
                   [[0.9, 1.0], [1.1, 1.2]]]
)

# Transpose dimensions
transposed_z = batch_z.rearrange("batch dim -> dim batch")
print(f"Original shape: {batch_z.shape}, Transposed shape: {transposed_z.shape}")

# Reshape
flattened_z = batch_z.rearrange("batch dim -> (batch dim)")
print(f"Flattened shape: {flattened_z.shape}")
```

### Repetition

```python
# Repeat a zonotope
repeated_z = z.repeat("dim -> repeat dim", repeat=3)
print(f"Original shape: {z.shape}, Repeated shape: {repeated_z.shape}")
```

### Summation

```python
# Sum along a dimension
summed_z = batch_z.sum(dim=0)
print(f"Original shape: {batch_z.shape}, Summed shape: {summed_z.shape}")
```

## Device and Type Conversion

Convert zonotopes between devices and data types:

```python
# Convert to float64
double_z = z.to(dtype=t.float64)
print(f"Original dtype: {z.dtype}, New dtype: {double_z.dtype}")

# Move to GPU (if available)
if t.cuda.is_available():
    gpu_z = z.to(device=t.device("cuda"))
    print(f"Original device: {z.device}, New device: {gpu_z.device}")
```

## Slicing and Indexing

Zonotopes support various indexing and slicing operations:

```python
# Create a batched zonotope
batch_z = Zonotope.from_values(
    center=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    infinity_terms=[[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
                   [[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]]]
)

# Single index
first_item = batch_z[0]
print(f"First item shape: {first_item.shape}")

# Slice
first_two = batch_z[:2]
print(f"First two shape: {first_two.shape}")

# Multi-dimensional slice
subset = batch_z[0, :2]
print(f"Subset shape: {subset.shape}")

# Using ellipsis
last_dim = batch_z[..., 0]
print(f"Last dim shape: {last_dim.shape}")
```

## Handling Error Terms

```python
# Expand infinity error terms
z_expanded = z.clone()
original_ei = z_expanded.Ei
z_expanded.expand_infinity_error_terms(5)
print(f"Original terms: {original_ei}, Expanded terms: {z_expanded.Ei}")

# Ensure zeros are updated (normally handled automatically)
z.update_zeros()
```

