from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt
import torch as t
from matplotlib.patches import Polygon
from numpy.typing import ArrayLike
from scipy.spatial import ConvexHull
from torch import Tensor

from zonotope.classical.z import Zonotope

LIGHTENING = "33"

ORANGE = "#FFB84C"
ORANGE_LIGHT = "#FFB84C" + LIGHTENING
VIOLET = "#A459D1"
VIOLET_LIGHT = "#A459D1" + LIGHTENING
PINK = "#F266AB"
PINK_LIGHT = "#F266AB" + LIGHTENING
TURQUOISE = "#2CD3E1"
TURQUOISE_LIGHT = "#2CD3E1" + LIGHTENING


def plot_zonotope_2d(
    z1: Zonotope,
    z2: Zonotope,
    n_samples: int = 10000,
    title: Optional[str] = None,
    alpha: float = 0.5,
    math_style: bool = True,
    arrow_size: float = 0.05,
    savefile: Optional[str] = None,
    concrete_range: Optional[ArrayLike] = None,
    concrete_values: Optional[ArrayLike] = None,
    set_aspect_equal: bool = True,
) -> None:
    assert z1.N == 2, "This function only works for 2D zonotopes, z1 is not 2D"
    assert z2.N == 2, "This function only works for 2D zonotopes, z2 is not 2D"

    _, ax = plt.subplots()

    samples_1 = (
        z1.sample_point(n_samples=n_samples, use_binary_weights=True)
        .detach()
        .cpu()
        .numpy()
    )
    samples_2 = (
        z2.sample_point(n_samples=n_samples, use_binary_weights=True)
        .detach()
        .cpu()
        .numpy()
    )

    if concrete_range is not None and concrete_values is not None:
        ax.plot(concrete_range, concrete_values, color="black")

    ax.scatter(
        samples_1[:, 0],
        samples_1[:, 1],
        s=1,
        alpha=0,
        c=ORANGE,
    )

    hull_1 = ConvexHull(samples_1)
    hull_points_1 = samples_1[hull_1.vertices]
    hull_polygon_1 = Polygon(
        hull_points_1,
        alpha=alpha,
        fill=True,
        edgecolor=ORANGE,
        facecolor=ORANGE_LIGHT,
    )

    ax.scatter(
        samples_2[:, 0],
        samples_2[:, 1],
        s=1,
        alpha=0,
        c=ORANGE,
    )

    hull_2 = ConvexHull(samples_2)
    hull_points_2 = samples_2[hull_2.vertices]
    hull_polygon_2 = Polygon(
        hull_points_2,
        alpha=alpha,
        fill=True,
        edgecolor=ORANGE,
        facecolor=ORANGE_LIGHT,
    )

    ax.add_patch(hull_polygon_1)
    ax.add_patch(hull_polygon_2)

    if title:
        ax.set_title(title)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if math_style:
        # Remove top and right spines to show only x and y axes
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        # Set ticks only on left and bottom
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")

        # Determine the axis limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Add arrows at the end of x and y axes
        ax.annotate(
            "",
            xytext=(xlim[0], ylim[0]),
            xy=(xlim[1] + arrow_size, ylim[0]),
            arrowprops=dict(arrowstyle="->"),
        )
        ax.annotate(
            "",
            xytext=(xlim[0], ylim[0]),
            xy=(xlim[0], ylim[1] + arrow_size),
            arrowprops=dict(arrowstyle="->"),
        )

        # Adjust limits to make arrows visible
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # No grid in math style
        ax.grid(False)

    else:
        ax.grid(True)

    if set_aspect_equal:
        ax.set_aspect("equal")
    if savefile is not None:
        plt.savefig(savefile)
    plt.show()


def plot_abstract_transformer(
    abstract_transformer: Callable[[Zonotope], Zonotope],
    concrete_function: Callable[[Tensor], Tensor],
    abstract_lims: Tuple[float, float, float],
    concrete_lims: Tuple[float, float],
    concrete_precision: int = 100,
    **kwargs,
):
    center_1 = (abstract_lims[0] + abstract_lims[1]) / 2
    center_2 = (abstract_lims[1] + abstract_lims[2]) / 2
    width_1 = (abstract_lims[1] - abstract_lims[0]) / 2
    width_2 = (abstract_lims[2] - abstract_lims[1]) / 2
    z_1 = Zonotope.from_values([center_1], [[width_1]])
    z_2 = Zonotope.from_values([center_2], [[width_2]])
    r_1 = abstract_transformer(z_1)
    r_2 = abstract_transformer(z_2)

    x = (
        t.arange(
            concrete_lims[0] * concrete_precision,
            concrete_lims[1] * concrete_precision,
        )
        / concrete_precision
    )
    y = concrete_function(x)

    zz1 = Zonotope.from_values(
        [center_1, r_1.W_C[0]], [[width_1, 0], r_1.W_G.tolist()[0]]
    )
    zz2 = Zonotope.from_values(
        [center_2, r_2.W_C[0]], [[width_2, 0], r_2.W_G.tolist()[0]]
    )

    plot_zonotope_2d(
        zz1,
        zz2,
        concrete_range=x.detach().cpu().numpy(),
        concrete_values=y.detach().cpu().numpy(),
        **kwargs,
    )
