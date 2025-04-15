from typing import List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch as t
from torch import Tensor

from zonotope.functional import relu
from zonotope.nn.main import linear
from zonotope.plot.theme import ORANGE, PINK, TURQUOISE, VIOLET
from zonotope.zonotope import Zonotope


def verify_network_robustness(
    model: t.nn.Module,
    input_point: Tensor,
    epsilon: float,
    norm_type: int = 2,
    target_class: int = 0,
) -> Tuple[bool, float]:
    """
    Verify robustness of a neural network around an input point.

    Args:
        model: The neural network to verify
        input_point: The center point to verify around
        epsilon: The perturbation radius
        norm_type: The type of norm for the perturbation (1, 2, or float('inf'))
        target_class: The expected correct class

    Returns:
        is_robust: Whether the network is robust around the input point
        margin: The minimum margin between the target class and other classes
    """
    # Create a zonotope representing the perturbed input region
    center = input_point.clone()

    # For ℓ∞ norm, we use infinity_terms
    if norm_type == float("inf"):
        infinity_terms = t.eye(input_point.shape[0]) * epsilon
        special_terms = None
    # For ℓp norms, we use special_terms
    else:
        infinity_terms = None
        special_terms = t.eye(input_point.shape[0]) * epsilon

    # Initialize the zonotope
    z = Zonotope(
        center=center,
        infinity_terms=infinity_terms,
        special_terms=special_terms,
        special_norm=norm_type,
    )

    # Propagate the zonotope through the network
    for layer in model.layers[:-1]:
        # Linear transformation
        z = linear(z, layer.weight, layer.bias)
        # ReLU activation
        z = relu(z)

    # Final layer (linear transformation only)
    z = linear(z, model.layers[-1].weight, model.layers[-1].bias)

    # Get the concrete bounds of the output zonotope
    lower_bound, upper_bound = z.concretize()

    # Calculate the worst-case margin between the target class and others
    margin = float("inf")
    for i in range(len(lower_bound)):
        if i == target_class:
            continue
        # Worst-case margin: min(target_class_lower - other_class_upper)
        current_margin = lower_bound[target_class] - upper_bound[i]
        margin = min(margin, current_margin.item())

    # The network is robust if the margin is positive
    is_robust = margin > 0

    return is_robust, margin


def compare_concrete_vs_abstract_propagation(
    model: t.nn.Module,
    input_point: Tensor,
    epsilon: float,
    norm_type: int = 2,
    num_samples: int = 1000,
):
    """
    Compare concrete points vs. abstract domain propagation

    Args:
        model: The neural network to analyze
        input_point: The center point
        epsilon: Perturbation radius
        norm_type: Type of norm (1, 2, or float('inf'))
        num_samples: Number of random samples to generate within the perturbation region
    """
    # Create a zonotope for the input region
    center = input_point.clone()

    if norm_type == float("inf"):
        infinity_terms = t.eye(input_point.shape[0]) * epsilon
        special_terms = None
    else:
        infinity_terms = None
        special_terms = t.eye(input_point.shape[0]) * epsilon

    input_zonotope = Zonotope(
        center=center,
        infinity_terms=infinity_terms,
        special_terms=special_terms,
        special_norm=norm_type,
    )

    sample_point = input_zonotope.sample_point(n_samples=num_samples)
    with t.no_grad():
        output = model(sample_point)

    # Propagate the zonotope through the network
    z = input_zonotope
    for _, layer in enumerate(model.layers[:-1]):
        z = linear(z, layer.weight, layer.bias)
        z = relu(z)
    z = linear(z, model.layers[-1].weight, model.layers[-1].bias)

    # Get bounds from the output zonotope
    lower_bound, upper_bound = z.concretize()
    lower_bound, upper_bound = (
        lower_bound.detach().cpu().numpy(),
        upper_bound.detach().cpu().numpy(),
    )

    # Plot the results
    plt.figure(figsize=(10, 6))

    # Calculate min and max values for each output dimension
    sample_min = t.min(output, dim=0)[0].detach().cpu().numpy()
    sample_max = t.max(output, dim=0)[0].detach().cpu().numpy()

    # Plot the bounds for each output dimension
    x = np.arange(len(lower_bound))

    plt.errorbar(
        x,
        sample_min + (sample_max - sample_min) / 2,
        yerr=(sample_max - sample_min) / 2,
        fmt="o",
        color=ORANGE,
        label="Sampled Points Range",
        capsize=5,
        markersize=8,
        elinewidth=2,
    )

    plt.errorbar(
        x + 0.1,
        (lower_bound + upper_bound) / 2,
        yerr=(upper_bound - lower_bound) / 2,
        fmt="s",
        color=VIOLET,
        label="Zonotope Bounds",
        capsize=5,
        markersize=8,
        elinewidth=2,
    )

    plt.xlabel("Output Dimension", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.title(
        f"Comparison of Concrete Samples vs. Zonotope Bounds (ℓ_{norm_type} norm)",
        fontsize=14,
    )
    plt.legend(frameon=True, facecolor="white", edgecolor=PINK, framealpha=0.8)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()

    # Print the precision gap
    zonotope_width = upper_bound - lower_bound
    sample_width = t.tensor(sample_max - sample_min)
    precision_gap = (zonotope_width / sample_width).mean().item()

    print(f"Average precision gap (zonotope width / sample width): {precision_gap:.4f}")

    return lower_bound, upper_bound, sample_min, sample_max


def visualize_decision_boundary(
    model: t.nn.Module,
    xlim: Tuple[float, float] = (-3.0, 3.0),
    ylim: Tuple[float, float] = (-3.0, 3.0),
    grid_size: int = 100,
    norm: Literal["inf", "2"] = "2",
    verified_points: Optional[List[Tuple[Tensor, float, bool]]] = None,
) -> None:
    """
    Visualize the decision boundary of a 2D model with verified regions

    Args:
        model: Neural network with 2D input
        xlim: x-axis limits
        ylim: y-axis limits
        grid_size: Resolution of the grid
        verified_points: List of (point, epsilon, is_robust) tuples for visualization
    """
    # Create a grid of points
    x = np.linspace(xlim[0], xlim[1], grid_size)
    y = np.linspace(ylim[0], ylim[1], grid_size)
    X, Y = np.meshgrid(x, y)

    # Evaluate the model on the grid
    with t.no_grad():
        Z = np.zeros_like(X)
        for i in range(grid_size):
            for j in range(grid_size):
                point = t.tensor([X[i, j], Y[i, j]], dtype=t.float32)
                output = model(point)
                Z[i, j] = output.argmax().item()

    # Plot the decision boundary
    # Plot the decision boundary
    plt.figure(figsize=(10, 8))
    cmap = plt.cm.colors.ListedColormap(
        [ORANGE, VIOLET, PINK, TURQUOISE][: model.layers[-1].out_features]
    )
    plt.contourf(X, Y, Z, alpha=0.6, cmap=cmap)
    plt.colorbar(ticks=np.arange(model.layers[-1].out_features))

    # Plot verified points if provided
    if verified_points:
        for point, epsilon, is_robust in verified_points:
            x, y = point[0].item(), point[1].item()
            color = TURQUOISE if is_robust else PINK

            if epsilon > 0:
                if norm == "2":
                    # For L2 norm: plot a circle
                    circle = plt.Circle(
                        (x, y), epsilon, color=color, fill=False, linewidth=2, alpha=0.7
                    )
                    plt.gca().add_patch(circle)
                elif norm == "inf":
                    # For L∞ norm: plot a rectangle
                    rectangle = plt.Rectangle(
                        (x - epsilon, y - epsilon),
                        2 * epsilon,
                        2 * epsilon,
                        color=color,
                        fill=False,
                        linewidth=2,
                        alpha=0.7,
                    )
                    plt.gca().add_patch(rectangle)

            # Plot the center point
            plt.scatter(x, y, color=VIOLET, s=50, edgecolor=ORANGE, linewidth=1.5)

    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.grid(True, linestyle="--", alpha=0.7, color="gray")
    plt.title("Model Decision Boundary with Verified Regions", fontsize=14)
    plt.xlabel("x₁", fontsize=12)
    plt.ylabel("x₂", fontsize=12)
    plt.show()
