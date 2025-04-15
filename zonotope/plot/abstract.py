from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

from zonotope.utils import sample_infinite
from zonotope.zonotope import Zonotope


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
) -> None:
    assert z.N == 2, "This function only works for 2D zonotopes"

    _, ax = plt.subplots()

    samples = z.sample_point(n_samples=n_samples, binary=True).detach().cpu().numpy()

    ax.scatter(
        samples[:, 0],
        samples[:, 1],
        s=1,
        alpha=alpha if plot_density_points else 0.0,
        c="blue",
    )

    if plot_convex_hull and samples.shape[0] > 3:
        hull = ConvexHull(samples)
        hull_points = samples[hull.vertices]
        hull_polygon = Polygon(
            hull_points,
            alpha=alpha,
            fill=True,
            edgecolor="blue",
            facecolor="lightblue",
        )
        ax.add_patch(hull_polygon)

    if plot_without_special_terms:
        samples_inf = sample_infinite(z).detach().cpu().numpy()

        ax.scatter(
            samples_inf[:, 0],
            samples_inf[:, 1],
            s=1,
            alpha=alpha if plot_density_points else 0.0,
            c="red",
        )

        if plot_convex_hull and samples_inf.shape[0] > 3:
            hull_inf = ConvexHull(samples_inf)
            hull_points_inf = samples_inf[hull_inf.vertices]
            hull_polygon_inf = Polygon(
                hull_points_inf,
                alpha=alpha,
                fill=True,
                edgecolor="green",
                facecolor="lightgreen",
            )
            ax.add_patch(hull_polygon_inf)

    if plot_center:
        center = z.W_C.detach().cpu().numpy()
        ax.scatter(center[0], center[1], c="red", s=100, marker="*")

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

    ax.set_aspect("equal")
    if savefile is not None:
        plt.savefig(savefile)
    plt.show()
