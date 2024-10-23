"""Module for plotting functions."""
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import tifffile
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


def save_current_fig(
    graphics_output_format: List[str],
    output_path: str,
    filename: str,
    filename_suffix: str,
    image: Optional[np.ndarray] = None,
):
    """Save the current figure to the output path with the given filename and suffix."""
    # prevent text outside figure area
    plt.tight_layout()

    filename = str(filename)
    filename_suffix = str(filename_suffix)

    Path(output_path).mkdir(parents=True, exist_ok=True)

    if "pdf" in graphics_output_format:
        plt.savefig(
            str(Path(output_path).joinpath(filename + filename_suffix + ".pdf"))
        )
    if "svg" in graphics_output_format:
        plt.savefig(
            str(Path(output_path).joinpath(filename + filename_suffix + ".svg"))
        )
    if "png" in graphics_output_format:
        plt.savefig(
            str(Path(output_path).joinpath(filename + filename_suffix + ".png"))
        )
    if "tif" in graphics_output_format and image is not None:
        tifffile.imwrite(
            str(Path(output_path).joinpath(filename + filename_suffix + ".tif")), image
        )


def add_vector(
    ax, x_pos_p1, y_pos_p1, x_pos_p2, y_pos_p2, markersize=2, font_color="w"
):
    """Add a vector to the plot."""
    ax.plot(x_pos_p1, y_pos_p1, ".g", markersize=markersize)
    ax.plot(x_pos_p2, y_pos_p2, ".m", markersize=markersize)
    ax.arrow(
        x_pos_p1,
        y_pos_p1,
        x_pos_p2 - x_pos_p1,
        y_pos_p2 - y_pos_p1,
        color=font_color,
        width=4,
    )


def add_colorbar(fig, cax, ax, yticks, label):
    """Add a colorbar to the plot."""
    color_bar = fig.colorbar(cax, ax=ax, shrink=0.3)
    color_bar.set_label(label)
    color_bar.ax.set_yticks(yticks)


def add_scalebar(
    ax, length_scalebar_microns, pixel_to_micron_ratio, size_vertical, font_color="w"
):
    """Add a scalebar to the plot."""
    length_scalebar_pixels = length_scalebar_microns * pixel_to_micron_ratio

    if sys.getdefaultencoding() == "utf-8":
        entity = "\u03BC"
    else:
        entity = "mu"
    text = f"{length_scalebar_microns} {entity}m"

    scalebar = AnchoredSizeBar(
        ax.transData,
        length_scalebar_pixels,
        text,
        "lower right",
        pad=0.1,
        color=font_color,
        frameon=False,
        size_vertical=size_vertical,
    )
    ax.add_artist(scalebar)


def add_title(ax, plot_title: str, im_junction: np.ndarray, axis_on: bool):
    """Add a title to the plot."""
    ax.set_title(plot_title)
    ax.set_xlim(0, im_junction.shape[1])  # x-axis is second dimension in numpy array
    ax.set_ylim(0, im_junction.shape[0])
    ax.invert_yaxis()
    if not axis_on:
        ax.axis("off")
