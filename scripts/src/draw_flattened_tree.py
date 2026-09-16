import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


def plot_flattened_tree(fig, ax, alphas, save_path, file_name):

    segments = []
    seg_volumes = []

    node_positions = []
    node_volumes = []

    LINE_WIDTH = 1.8

    def recurse(g, i, x0, y0, angle, angle_range, radius_step):

        vol = alphas[g][i]

        x1 = x0 + radius_step * np.cos(angle)
        y1 = y0 + radius_step * np.sin(angle)

        node_positions.append((x1, y1))
        node_volumes.append(vol)

        segments.append([(x0, y0), (x1, y1)])
        seg_volumes.append(vol)

        if g + 1 < len(alphas):

            recurse(
                g + 1, 2 * i,
                x1, y1,
                angle - angle_range / 4,
                angle_range / 2,
                radius_step
            )

            recurse(
                g + 1, 2 * i + 1,
                x1, y1,
                angle + angle_range / 4,
                angle_range / 2,
                radius_step
            )

    recurse(
        0, 0,
        x0=0,
        y0=-1.0,
        angle=np.pi / 2,
        angle_range=2 * np.pi,
        radius_step=1.0
    )

    segments = np.asarray(segments)
    seg_volumes = np.asarray(seg_volumes, dtype=float)

    node_positions = np.asarray(node_positions)
    node_volumes = np.asarray(node_volumes, dtype=float)

    norm = Normalize(vmin=0, vmax=0.5)

    nan_mask = np.isnan(seg_volumes)
    valid_mask = ~nan_mask

    lc_lines = LineCollection(
        segments[valid_mask],
        linewidths=LINE_WIDTH,
        colors="grey",
        capstyle="round",
        alpha=0.95
    )

    ax.add_collection(lc_lines)

    lc_nan = LineCollection(
        segments[nan_mask],
        linewidths=LINE_WIDTH,
        colors="white",
        capstyle="round",
        alpha=0.95
    )

    ax.add_collection(lc_nan)

    def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
        new_cmap = mcolors.LinearSegmentedColormap.from_list(
            f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
            cmap(np.linspace(minval, maxval, n))
        )
        return new_cmap

    cmap = plt.get_cmap('inferno')
    cut_cmap = truncate_colormap(cmap, 0., 0.95)
    cut_cmap.set_bad("white")

    sc = ax.scatter(
        node_positions[:, 0],
        node_positions[:, 1],
        c=node_volumes,
        cmap=cut_cmap,
        norm=norm,
        s=100,
        zorder=3
    )

    cbar = fig.colorbar(
        sc,
        ax=ax,
        shrink=0.6,
        pad=0.02
    )

    cbar.set_label(
        "Splitting factor",
        fontsize=20,
        labelpad=10
    )

    cbar.ax.tick_params(
        labelsize=17,
        length=8,
        width=1.5
    )

    ax.autoscale()
    ax.set_aspect("equal")
    ax.axis("off")

    fig.tight_layout()

    fig.savefig(f"{save_path}/{file_name}.svg", dpi=500, format="svg")

    plt.show()