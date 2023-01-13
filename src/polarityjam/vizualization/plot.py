import math
from pathlib import Path

import cmocean as cm
import matplotlib as mpl
import numpy as np
import tifffile
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from polarityjam.polarityjam_logging import get_logger

# for figure plot resolution  # todo: parameters?
FIGURE_DPI = 300
FONTSIZE_TEXT_ANNOTATIONS = 3
MARKERSIZE = 2
ALPHA_MASKS = 0.5
CELL_OUTLINE_INTENSITY = 30


def save_current_fig(graphics_output_format, output_path, filename, filename_suffix, image=None):
    # prevent text outside figure area
    plt.tight_layout()

    filename = str(filename)
    filename_suffix = str(filename_suffix)

    if "pdf" in graphics_output_format:
        plt.savefig(str(Path(output_path).joinpath(filename + filename_suffix + ".pdf")))
    if "svg" in graphics_output_format:
        plt.savefig(str(Path(output_path).joinpath(filename + filename_suffix + ".svg")))
    if "png" in graphics_output_format:
        plt.savefig(str(Path(output_path).joinpath(filename + filename_suffix + ".png")))
    if "tif" in graphics_output_format and image is not None:
        tifffile.imwrite(str(Path(output_path).joinpath(filename + filename_suffix + ".tif")), image)


def set_figure_dpi():
    mpl.rcParams['figure.dpi'] = FIGURE_DPI


def _add_single_cell_polarity_vector(ax, x_pos_p1, y_pos_p1, x_pos_p2, y_pos_p2):
    ax.plot(y_pos_p1, x_pos_p1, '.g', markersize=MARKERSIZE)
    ax.plot(y_pos_p2, x_pos_p2, '.m', markersize=MARKERSIZE)
    ax.arrow(
        y_pos_p1,
        x_pos_p1,
        y_pos_p2 - y_pos_p1,
        x_pos_p2 - x_pos_p1,
        color='white', width=4
    )


def _add_single_cell_eccentricity_axis(
        ax, y0, x0, orientation, major_axis_length, minor_axis_length, eccentricity
):
    x1_ma, x1_mi, x2_ma, x2_mi, y1_ma, y1_mi, y2_ma, y2_mi = _calc_single_cell_axis_orientation_vector(
        x0, y0, orientation, major_axis_length, minor_axis_length
    )

    ax.plot((y1_ma, y2_ma), (x1_ma, x2_ma), '--w', linewidth=0.5)
    ax.plot((y1_mi, y2_mi), (x1_mi, x2_mi), '--w', linewidth=0.5)
    ax.plot(y0, x0, '.b', markersize=MARKERSIZE)
    ax.text(y0, x0, str(np.round(eccentricity, 2)), color="yellow", fontsize=FONTSIZE_TEXT_ANNOTATIONS)


def _add_colorbar(fig, cax, ax, yticks, label):
    color_bar = fig.colorbar(cax, ax=ax, shrink=0.3)
    color_bar.set_label(label)
    color_bar.ax.set_yticks(yticks)


def _add_scalebar(ax, length_scalebar_microns, pixel_to_micron_ratio, size_vertical):
    length_scalebar_pixels = length_scalebar_microns / pixel_to_micron_ratio
    text = "%s mu m" % length_scalebar_microns

    scalebar = AnchoredSizeBar(
        ax.transData,
        length_scalebar_microns, text, 'lower right',
        pad=0.1,
        color='white',
        frameon=False,
        size_vertical=size_vertical
    )
    ax.add_artist(scalebar)


def _add_cell_eccentricity(fig, ax, im_junction, cell_mask, cell_eccentricity):
    v_min = 0.0
    v_max = 1.0
    yticks = [0.0, 0.5, 1.0]

    ax.imshow(im_junction, cmap=plt.cm.gray, alpha=1.0)

    # show cell_eccentricity everywhere but background label
    cax_0 = ax.imshow(
        np.ma.masked_where(cell_mask == 0, cell_eccentricity), cmap=plt.cm.bwr, vmin=v_min, vmax=v_max,
        alpha=0.5
    )

    # colorbar
    _add_colorbar(fig, cax_0, ax, yticks, "eccentricity")


def _add_nuclei_eccentricity(fig, ax, im_junction, nuclei_mask, nuclei_eccentricity):
    v_min = 0.0
    v_max = 1.0
    yticks = [0.0, 0.5, 1.0]

    ax.imshow(im_junction, cmap=plt.cm.gray, alpha=1.0)

    # show nuclei eccentricity everywhere but background label
    nuclei_mask_ = np.where(nuclei_mask == True, 1, 0)
    cax_1 = ax.imshow(np.ma.masked_where(
        nuclei_mask_ == 0, nuclei_eccentricity), cmap=plt.cm.bwr, vmin=v_min, vmax=v_max, alpha=0.5
    )

    # colorbar
    _add_colorbar(fig, cax_1, ax, yticks, "eccentricity")


def _add_title(ax, plot_title, im_junction, axis_on):
    ax.set_title(plot_title)
    ax.set_xlim(0, im_junction.shape[1])
    ax.set_ylim(0, im_junction.shape[0])
    ax.invert_yaxis()
    if not axis_on:
        ax.axis('off')


def _calc_nuc_eccentricity(single_cell_props, cell_mask, nuclei_mask):
    nuclei_eccentricity = np.zeros((cell_mask.shape[0], cell_mask.shape[1]))
    for index, row in single_cell_props.iterrows():
        row_label = int(row['label'])

        # exclude background
        if row_label == 0:
            continue
        single_cell_mask = np.where(cell_mask == row_label, True, 0)
        single_nuclei_mask_ = np.logical_and(single_cell_mask, nuclei_mask)
        nuclei_eccentricity += np.where(single_nuclei_mask_ == True, 1, 0) * row['nuc_eccentricity']

    get_logger().info("Maximal nuclei eccentricity: %s" % str(np.max(nuclei_eccentricity)))
    get_logger().info("Minimal nuclei eccentricity: %s" % str(np.min(nuclei_eccentricity)))

    return nuclei_eccentricity


def _calc_single_cell_axis_orientation_vector(x, y, orientation, major_axis_length, minor_axis_length):
    x1_major = x + math.sin(orientation) * 0.5 * major_axis_length
    y1_major = y - math.cos(orientation) * 0.5 * major_axis_length
    x2_major = x - math.sin(orientation) * 0.5 * major_axis_length
    y2_major = y + math.cos(orientation) * 0.5 * major_axis_length

    x1_minor = x - math.cos(orientation) * 0.5 * minor_axis_length
    y1_minor = y - math.sin(orientation) * 0.5 * minor_axis_length
    x2_minor = x + math.cos(orientation) * 0.5 * minor_axis_length
    y2_minor = y + math.sin(orientation) * 0.5 * minor_axis_length

    return [x1_major, x1_minor, x2_major, x2_minor, y1_major, y1_minor, y2_major, y2_minor]


def _calc_nuc_orientation(single_cell_props, cell_mask, nuclei_mask):
    get_logger().info("Calculating nuclei orientation...")
    nuclei_orientation = np.zeros((cell_mask.shape[0], cell_mask.shape[1]))
    for index, row in single_cell_props.iterrows():
        row_label = int(row['label'])

        # exclude background
        if row_label == 0:
            continue
        single_cell_mask = np.where(cell_mask == row_label, True, 0)
        single_nuclei_mask_ = np.logical_and(single_cell_mask, nuclei_mask)
        nuclei_orientation += np.where(single_nuclei_mask_ == True, 1, 0) * row[
            'nuc_shape_orientation_rad'] * 180.0 / np.pi

    get_logger().info("Maximal nuclei orientation: %s" % str(np.max(nuclei_orientation)))
    get_logger().info("Minimal nuclei orientation: %s" % str(np.min(nuclei_orientation)))

    return nuclei_orientation


def _add_cell_orientation(fig, ax, im_junction, cell_mask, cell_orientation):
    v_min = 0.0
    v_max = 180.0
    yticks = [0.0, 45.0, 90.0, 135.0, 180.0]

    ax.imshow(im_junction, cmap=plt.cm.gray, alpha=1.0)

    # show cell_orientation everywhere but background label
    cax = ax.imshow(
        np.ma.masked_where(cell_mask == 0, cell_orientation), cmap=cm.cm.phase, vmin=v_min, vmax=v_max,
        alpha=0.75
    )

    # colorbar
    _add_colorbar(fig, cax, ax, yticks, "shape orientation (degree)")


def _add_nuclei_orientation(fig, ax, im_junction, nuclei_mask, nuclei_orientation):
    v_min = 0.0
    v_max = 180.0
    yticks = [0.0, 45.0, 90.0, 135.0, 180.0]

    ax.imshow(im_junction, cmap=plt.cm.gray, alpha=1.0)

    # show nuclei orientation everywhere but background label
    nuclei_mask_ = np.where(nuclei_mask == True, 1, 0)
    cax_1 = ax.imshow(np.ma.masked_where(
        nuclei_mask_ == 0, nuclei_orientation), cmap=cm.cm.phase, vmin=v_min, vmax=v_max, alpha=0.75
    )

    # colorbar
    _add_colorbar(fig, cax_1, ax, yticks, "shape orientation (degree)")


def _add_single_cell_orientation_degree_axis(
        ax, y0, x0, orientation, major_axis_length, minor_axis_length,
):
    x1_ma, x1_mi, x2_ma, x2_mi, y1_ma, y1_mi, y2_ma, y2_mi = _calc_single_cell_axis_orientation_vector(
        x0, y0, orientation, major_axis_length, minor_axis_length
    )
    orientation_degree = 180.0 * orientation / np.pi

    ax.plot((y1_ma, y2_ma), (x1_ma, x2_ma), '--w', linewidth=0.5)
    ax.plot((y1_mi, y2_mi), (x1_mi, x2_mi), '--w', linewidth=0.5)
    ax.plot(y0, x0, '.b', markersize=MARKERSIZE)
    ax.text(y0, x0, str(int(np.round(orientation_degree, 0))), color="yellow", fontsize=FONTSIZE_TEXT_ANNOTATIONS)
