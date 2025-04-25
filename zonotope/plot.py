from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt
import torch as t
from matplotlib.patches import Polygon
from numpy.typing import ArrayLike
from scipy.spatial import ConvexHull
from torch import Tensor

from zonotope.zonotope import Zonotope

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
    z: Zonotope,
    n_samples: int = 10000,
    plot_convex_hull: bool = True,
    plot_density_points: bool = False,
    plot_without_special_terms: bool = False,
    plot_center: bool = False,
    title: Optional[str] = None,
    alpha: float = 0.5,
    math_style: bool = True,
    arrow_size: float = 0.05,
    savefile: Optional[str] = None,
    concrete_range: Optional[ArrayLike] = None,
    concrete_values: Optional[ArrayLike] = None,
    set_aspect_equal: bool = True,
) -> None:
    assert z.N == 2, "This function only works for 2D zonotopes"

    _, ax = plt.subplots()

    samples = (
        z.sample_point(n_samples=n_samples, use_binary_weights=True)
        .detach()
        .cpu()
        .numpy()
    )

    if concrete_range is not None and concrete_values is not None:
        ax.plot(concrete_range, concrete_values, color="black")

    ax.scatter(
        samples[:, 0],
        samples[:, 1],
        s=1,
        alpha=alpha if plot_density_points else 0.0,
        c=ORANGE,
    )

    if plot_convex_hull and samples.shape[0] > 3:
        hull = ConvexHull(samples)
        hull_points = samples[hull.vertices]
        hull_polygon = Polygon(
            hull_points,
            alpha=alpha,
            fill=True,
            edgecolor=ORANGE,
            facecolor=ORANGE_LIGHT,
        )
        ax.add_patch(hull_polygon)

    if plot_without_special_terms:
        samples_inf = (
            z.sample_point(
                n_samples=n_samples,
                use_binary_weights=True,
                include_special_terms=False,
            )
            .detach()
            .cpu()
            .numpy()
        )

        ax.scatter(
            samples_inf[:, 0],
            samples_inf[:, 1],
            s=1,
            alpha=alpha if plot_density_points else 0.0,
            c=PINK,
        )

        if plot_convex_hull and samples_inf.shape[0] > 3:
            hull_inf = ConvexHull(samples_inf)
            hull_points_inf = samples_inf[hull_inf.vertices]
            hull_polygon_inf = Polygon(
                hull_points_inf,
                alpha=alpha,
                fill=True,
                edgecolor=PINK,
                facecolor=PINK_LIGHT,
            )
            ax.add_patch(hull_polygon_inf)

    if plot_center:
        center = z.W_C.detach().cpu().numpy()
        ax.scatter(center[0], center[1], c=PINK, s=100, marker="*")

    if title:
        ax.set_title(title)
    else:
        p_str = "inf" if z.p == float("inf") else str(z.p)
        ax.set_title(f"2D Zonotope (p={p_str}, Es={z.Es}, Ei={z.Ei})")

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
    abstract_lims: Tuple[float, float],
    concrete_lims: Tuple[float, float],
    concrete_precision: int = 100,
    **kwargs,
):
    center = (abstract_lims[0] + abstract_lims[1]) / 2
    width = (abstract_lims[1] - abstract_lims[0]) / 2
    z = Zonotope.from_values([center], [[width]])
    r = abstract_transformer(z)

    x = (
        t.arange(
            concrete_lims[0] * concrete_precision,
            concrete_lims[1] * concrete_precision,
        )
        / concrete_precision
    )
    y = concrete_function(x)

    zz = Zonotope.from_values([center, r.W_C[0]], [[width, 0], r.W_Ei.tolist()[0]])

    plot_zonotope_2d(
        zz,
        concrete_range=x.detach().cpu().numpy(),
        concrete_values=y.detach().cpu().numpy(),
        **kwargs,
    )
