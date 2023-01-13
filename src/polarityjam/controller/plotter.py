import json
import math
import os

import cmocean as cm
import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from polarityjam.model.collection import PropertiesCollection
from polarityjam.model.masks import get_single_cell_mask, get_outline_from_mask, get_single_cell_nuc_mask
from polarityjam.model.parameter import PlotParameter, ImageParameter
from polarityjam.polarityjam_logging import get_logger
from polarityjam.vizualization.plot import _add_single_cell_polarity_vector, \
    _add_title, \
    save_current_fig, _add_cell_eccentricity, \
    _calc_nuc_eccentricity, _add_nuclei_eccentricity, _add_single_cell_eccentricity_axis, _add_cell_orientation, \
    _calc_nuc_orientation, _add_nuclei_orientation, _add_single_cell_orientation_degree_axis, _add_scalebar, \
    _add_colorbar


class Plotter:

    def __init__(self, params: PlotParameter):
        self.params = params
        Plotter.set_figure_dpi(self.params.dpi)

    @staticmethod
    def set_figure_dpi(figure_dpi):
        matplotlib.rcParams['figure.dpi'] = figure_dpi

    def _get_figure(self, n_subfigures):
        w, h = self.params.graphics_width, self.params.graphics_height
        fig, ax = plt.subplots(1, n_subfigures, figsize=(w * n_subfigures, h))
        plt.tight_layout()

        return fig, ax

    def _get_polarity_angle_mask(self, cell_mask, collection, img_name, feature):  # todo: place in extractor?
        cell_angle = np.zeros((cell_mask.shape[0], cell_mask.shape[1]))
        for index, row in collection.get_properties_by_img_name(img_name).iterrows():
            row_label = int(row['label'])
            if row_label == 0:
                continue
            cell_angle += get_single_cell_mask(cell_mask, row_label) * row[feature]
        polarity_angle = np.ma.masked_where(cell_mask == 0, cell_angle)

        get_logger().info("Maximal %s: %s" % (feature, str(np.max(polarity_angle))))
        get_logger().info("Minimal %s: %s" % (feature, str(np.min(polarity_angle))))

        return polarity_angle

    def _get_outlines(self, im_marker, cell_mask, nuclei_mask, single_cell_dataset):
        outlines_cell = np.zeros((im_marker.shape[0], im_marker.shape[1]))
        outlines_mem = np.copy(outlines_cell)
        outlines_nuc = np.copy(outlines_cell)

        for cell_label in single_cell_dataset["label"]:
            feature_row = single_cell_dataset.loc[single_cell_dataset["label"] == cell_label]
            intensity_cell = feature_row["marker_mean_expression"].values[0]
            intensity_mem = feature_row["marker_mean_expression_mem"].values[0]
            intensity_nuc = None

            if nuclei_mask is not None:
                intensity_nuc = feature_row["marker_mean_expression_nuc"].values[0]

            single_cell_mask = get_single_cell_mask(cell_mask, cell_label)
            single_cell_mask[single_cell_mask > 0] = intensity_cell
            outline_cell = get_outline_from_mask(single_cell_mask, self.params.outline_width)
            single_cell_mask_ = np.where(outline_cell == True, 0, single_cell_mask)
            outlines_cell += single_cell_mask_

            outline_mem = get_outline_from_mask(single_cell_mask, self.params.membrane_thickness)
            outline_mem_ = np.where(outline_mem == True, intensity_mem, 0)
            outlines_mem += outline_mem_

            # nuclei marker intensity
            if nuclei_mask is not None:
                single_nucleus_mask = get_single_cell_nuc_mask(nuclei_mask, cell_mask, cell_label)
                single_nucleus_mask[single_nucleus_mask > 0] = intensity_nuc
                outline_nuc = get_outline_from_mask(single_nucleus_mask, self.params.outline_width)
                single_nuc_mask_ = np.where(outline_nuc == True, 0, single_nucleus_mask)
                outlines_nuc += single_nuc_mask_

        return [outlines_cell, outlines_mem, outlines_nuc]

    def _masked_cell_outlines(self, img, cell_mask):
        # cell outlines
        outlines_cells = np.zeros((img.shape[0], img.shape[1]))
        for cell_label in np.unique(cell_mask):
            # exclude background
            if cell_label == 0:
                continue

            single_cell_mask = get_single_cell_mask(cell_mask, cell_label)
            outline_cell = get_outline_from_mask(single_cell_mask, self.params.outline_width)
            outlines_cells = np.logical_or(outlines_cells, outline_cell)

        # convert cell outlines to image
        outlines_cells_rgba = np.where(outlines_cells == True, 255, 0)
        outlines_cells_rgba = np.dstack([outlines_cells_rgba] * 3)
        outlines_cells_rgba_masked = np.ma.masked_where(np.dstack([outlines_cells] * 3) == False, outlines_cells_rgba)

        return outlines_cells_rgba_masked

    def plot_channels(self, seg_img, seg_img_params: ImageParameter, output_path, filename, close=False):
        """Plots the separate channels from the input file given."""
        get_logger().info("Plotting: input channels")

        filename, _ = os.path.splitext(os.path.basename(filename))

        if seg_img_params.channel_junction is not None and seg_img_params.channel_nucleus is not None:
            fig, ax = self._get_figure(2)

            # junction channel
            c_junction = seg_img_params.channel_junction
            ax[0].imshow(seg_img[c_junction, :, :])
            _add_title(
                ax[0], "junction channel", seg_img[c_junction, :, :], self.params.show_graphics_axis
            )

            # nucleus channel
            c_nucleus = seg_img_params.channel_nucleus
            ax[1].imshow(seg_img[c_nucleus, :, :])
            _add_title(
                ax[1], "nuclei channel", seg_img[c_nucleus, :, :], self.params.show_graphics_axis
            )

            axes = [ax[0], ax[1]]
        else:
            fig, ax = self._get_figure(1)

            # first channel
            ax.imshow(seg_img[:, :])
            _add_title(ax, "first channel", seg_img[:, :], self.params.show_graphics_axis)
            axes = [ax]

        self._finish_plot(fig, output_path, filename, "_channels", axes, close)

    def plot_mask(self, mask, seg_img, seg_img_params, output_path, filename, close=False):
        """Plots the segmentation mask, together with the separate channels from the input image."""
        get_logger().info("Plotting: segmentation masks")

        filename, _ = os.path.splitext(os.path.basename(filename))

        # color each cell differently
        cell_idx = np.unique(mask)
        cell_idx = np.delete(cell_idx, 0)
        mask_ = np.copy(mask)

        new_col = np.copy(cell_idx)
        np.random.seed(42)  # set seed for reproducibility
        np.random.shuffle(new_col)
        for i in range(len(cell_idx)):
            mask_[mask == cell_idx[i]] = new_col[i]

        # ignore background
        mask_ = np.where(mask > 0, mask_, np.nan)

        if seg_img_params.channel_junction is not None and seg_img_params.channel_nucleus is not None:
            fig, ax = self._get_figure(3)

            ax[0].imshow(seg_img[0, :, :])
            _add_title(ax[0], "junction channel", seg_img[0, :, :], self.params.show_graphics_axis)

            ax[1].imshow(seg_img[1, :, :])
            _add_title(ax[1], "nuclei channel", seg_img[1, :, :], self.params.show_graphics_axis)

            ax[2].imshow(seg_img[0, :, :])
            ax[2].imshow(mask_, cmap=plt.cm.gist_rainbow, alpha=0.5)
            _add_title(ax[2], "segmentation", seg_img[0, :, :], self.params.show_graphics_axis)

            axes = [ax[0], ax[1], ax[2]]
        else:
            fig, ax = self._get_figure(2)

            s_img = seg_img[:, :]

            # ax 1
            ax[0].imshow(s_img)
            _add_title(ax[0], "junction channel", s_img, self.params.show_graphics_axis)

            # ax 2
            ax[1].imshow(s_img)
            ax[1].imshow(mask_, cmap=plt.cm.gist_rainbow, alpha=0.5)
            _add_title(ax[1], "segmentation", s_img, self.params.show_graphics_axis)

            axes = [ax[0], ax[1]]

        self._finish_plot(fig, output_path, filename, "_segmentation", axes, close)

    def plot_organelle_polarity(self, collection, img_name, close=False):
        im_junction = collection.get_image_channel_by_img_name(img_name, "junction")
        cell_mask = collection.get_mask_by_img_name(img_name).cell_mask_rem_island
        nuclei_mask = collection.get_mask_by_img_name(img_name).nuclei_mask
        organelle_mask = collection.get_mask_by_img_name(img_name).organelle_mask

        get_logger().info("Plotting: organelle polarity")

        fig, ax = self._get_figure(1)

        # resources image
        ax.imshow(im_junction, cmap=plt.cm.gray, alpha=1.0)

        # determine polarity_angle
        polarity_angle = self._get_polarity_angle_mask(cell_mask, collection, img_name, "organelle_orientation_deg")

        # plot polarity angle
        cax = ax.imshow(polarity_angle, cmap=cm.cm.phase, vmin=0, vmax=360, alpha=0.5)
        color_bar = fig.colorbar(cax, ax=ax, shrink=0.3)  # , extend='both')
        color_bar.set_label("polarity angle")
        color_bar.ax.set_yticks([0, 90, 180, 270, 360])

        # plot differently colored organelle (red) and nuclei (blue)
        zero = np.zeros((im_junction.shape[0], im_junction.shape[1]))
        rgb_organelle = np.dstack(
            (organelle_mask.astype(int) * 256, zero, zero, organelle_mask.astype(float) * 0.5))
        rgb_nuclei = np.dstack((zero, zero, nuclei_mask.astype(int) * 256, nuclei_mask.astype(float) * 0.5))
        ax.imshow(rgb_nuclei)
        ax.imshow(rgb_organelle)

        # plot polarity vector
        for index, row in collection.get_properties_by_img_name(img_name).iterrows():
            _add_single_cell_polarity_vector(ax, row["nuc_X"], row["nuc_Y"], row["organelle_X"], row["organelle_Y"])
            if self.params.show_polarity_angles:
                ax.text(row["cell_Y"], row["cell_X"], str(int(np.round(row["organelle_orientation_deg"], 0))),
                        color="yellow", fontsize=6)

        # set title and ax limits
        _add_title(ax, "organelle orientation", im_junction, self.params.show_graphics_axis)

        self._finish_plot(fig, collection.get_out_path_by_name(img_name), img_name, "_nuclei_organelle_vector", [ax],
                          close, polarity_angle)

    def plot_nuc_displacement_orientation(self, collection, img_name, close=False):
        im_junction = collection.img_channel_dict[img_name]["junction"]
        cell_mask = collection.masks_dict[img_name].cell_mask_rem_island
        nuclei_mask = collection.masks_dict[img_name].nuclei_mask
        base_filename = img_name

        get_logger().info("Plotting: marker nucleus polarity")

        # figure and axes
        fig, ax = self._get_figure(1)

        # resources image
        ax.imshow(im_junction, cmap=plt.cm.gray, alpha=1.0)

        # determine nucleus polarity_angle
        nuc_polarity_angle = self._get_polarity_angle_mask(cell_mask, collection, img_name,
                                                           "nuc_displacement_orientation_deg")

        # plot polarity angle
        cax = ax.imshow(nuc_polarity_angle, cmap=cm.cm.phase, vmin=0, vmax=360, alpha=0.5)
        color_bar = fig.colorbar(cax, ax=ax, shrink=0.3)  # , extend='both')
        color_bar.set_label("polarity angle")
        color_bar.ax.set_yticks([0, 90, 180, 270, 360])

        # plot nuclei (blue)
        zero = np.zeros((im_junction.shape[0], im_junction.shape[1]))
        rgb_nuclei = np.dstack((zero, zero, nuclei_mask.astype(int) * 256, nuclei_mask.astype(float) * 0.5))
        ax.imshow(rgb_nuclei)

        # plot polarity vector
        for index, row in collection.get_properties_by_img_name(img_name).iterrows():
            _add_single_cell_polarity_vector(ax, row["cell_X"], row["cell_Y"], row["nuc_X"], row["nuc_Y"])
            if self.params.show_polarity_angles:
                ax.text(
                    row["nuc_Y"], row["nuc_X"], str(int(np.round(row["nuc_displacement_orientation_deg"], 0))),
                    color="yellow", fontsize=6
                )

        # set title and ax limits
        _add_title(ax, "nucleus displacement orientation", im_junction, self.params.show_graphics_axis)

        self._finish_plot(
            fig, collection.get_out_path_by_name(img_name), img_name, "_nucleus_displacement_orientation", [ax], close,
            nuc_polarity_angle
        )

    def plot_marker_expression(self, collection, img_name, close=False):
        im_marker = collection.img_channel_dict[img_name]["marker"]
        cell_mask = collection.masks_dict[img_name].cell_mask_rem_island
        single_cell_dataset = collection.dataset.loc[collection.dataset["filename"] == img_name]
        nuclei_mask = collection.masks_dict[img_name].nuclei_mask

        get_logger().info("Plotting: marker expression")
        # figure and axes
        number_sub_figs = 2  # mean intensity cell, mean intensity membrane
        if nuclei_mask is not None:
            nuclei_mask = nuclei_mask.astype(bool)
            number_sub_figs = 3  # (optional) mean intensity nucleus

        fig, ax = self._get_figure(number_sub_figs)

        # plot marker intensity for all subplots
        for i in range(number_sub_figs):
            ax[i].imshow(im_marker, cmap=plt.cm.gray, alpha=1.0)

        outlines_cell, outlines_mem, outlines_nuc = self._get_outlines(
            im_marker, cell_mask, nuclei_mask, single_cell_dataset
        )

        # cell and membrane outline
        outlines_cell_ = np.where(outlines_cell > 0, outlines_cell, np.nan)
        cax_1 = ax[0].imshow(outlines_cell_, plt.cm.bwr, alpha=0.5)

        outlines_mem_ = np.where(outlines_mem > 0, outlines_mem, np.nan)
        cax_2 = ax[1].imshow(outlines_mem_, plt.cm.bwr, alpha=0.5)

        # nuclei marker intensity
        cax_3 = None
        outlines_nuc_ = None
        if nuclei_mask is not None:
            outlines_nuc_ = np.where(outlines_nuc > 0, outlines_nuc, np.nan)
            cax_3 = ax[2].imshow(outlines_nuc_, plt.cm.bwr, alpha=0.75)  # always last axis

        # colorbar for cell
        yticks_cell = [np.nanmin(outlines_cell_), np.nanmax(outlines_cell_, )]
        _add_colorbar(fig, cax_1, ax[0], yticks_cell, "intensity cell")

        # colorbar for membrane
        yticks_mem = [np.nanmin(outlines_mem_), np.nanmax(outlines_mem_, )]
        _add_colorbar(fig, cax_2, ax[1], yticks_mem, "intensity membrane")

        # colorbar for nucleus
        if nuclei_mask is not None:
            yticks_nuc = [np.nanmin(outlines_nuc_), np.nanmax(outlines_nuc_)]
            _add_colorbar(fig, cax_3, ax[2], yticks_nuc, "intensity nucleus")

        # plot mean expression value of cell and membrane as text
        for index, row in single_cell_dataset.iterrows():
            ax[0].text(row["cell_Y"], row["cell_X"], str(np.round(row["marker_mean_expression"], 1)), color="w",
                       fontsize=7)
            ax[1].text(row["cell_Y"], row["cell_X"], str(np.round(row["marker_mean_expression_mem"], 1)), color="w",
                       fontsize=7)
            if nuclei_mask is not None:
                ax[2].text(
                    row["nuc_Y"], row["nuc_X"], str(np.round(row["marker_mean_expression_nuc"], 1)), color="w",
                    fontsize=7
                )

        # set title
        axes = [ax[0], ax[1]]
        _add_title(ax[0], "mean intensity cell", im_marker, self.params.show_graphics_axis)
        _add_title(ax[1], "mean intensity membrane", im_marker, self.params.show_graphics_axis)
        if nuclei_mask is not None:
            _add_title(ax[2], "mean intensity nucleus", im_marker, self.params.show_graphics_axis)
            axes = [ax[0], ax[1], ax[2]]

        self._finish_plot(fig, collection.get_out_path_by_name(img_name), img_name, "_marker_expression", axes, close)

    def plot_marker_polarity(self, collection, img_name, close=False):
        im_marker = collection.img_channel_dict[img_name]["marker"]
        cell_mask = collection.masks_dict[img_name].cell_mask_rem_island
        img_name = img_name

        get_logger().info("Plotting: marker polarity")

        # figure and axes
        fig, ax = self._get_figure(1)

        # plot marker intensity
        ax.imshow(im_marker, cmap=plt.cm.gray, alpha=1.0)

        # show cell outlines
        ax.imshow(self._masked_cell_outlines(im_marker, cell_mask), alpha=0.5)

        # add all polarity vectors
        for index, row in collection.get_properties_by_img_name(img_name).iterrows():
            _add_single_cell_polarity_vector(
                ax, row["cell_X"], row["cell_Y"], row["marker_centroid_X"], row["marker_centroid_Y"]
            )

        _add_title(ax, "marker polarity", im_marker, self.params.show_graphics_axis)

        self._finish_plot(fig, collection.get_out_path_by_name(img_name), img_name, "_marker_polarity", [ax], close)

    def plot_marker_nucleus_orientation(self, collection, img_name, close=False):
        im_junction = collection.img_channel_dict[img_name]["junction"]
        cell_mask = collection.masks_dict[img_name].cell_mask_rem_island
        nuclei_mask = collection.masks_dict[img_name].nuclei_mask
        output_path = collection.out_path_dict[img_name]

        get_logger().info("Plotting: marker nucleus polarity")

        # figure and axes
        fig, ax = self._get_figure(1)

        # resources image
        ax.imshow(im_junction, cmap=plt.cm.gray, alpha=1.0)

        # determine nucleus polarity_angle
        nuc_polarity_angle = self._get_polarity_angle_mask(cell_mask, collection, img_name,
                                                           "marker_nucleus_orientation_deg")

        # plot polarity angle
        cax = ax.imshow(nuc_polarity_angle, cmap=cm.cm.phase, vmin=0, vmax=360, alpha=0.5)
        color_bar = fig.colorbar(cax, ax=ax, shrink=0.3)  # , extend='both')
        color_bar.set_label("polarity angle")
        color_bar.ax.set_yticks([0, 90, 180, 270, 360])

        # plot nuclei (blue)
        zero = np.zeros((im_junction.shape[0], im_junction.shape[1]))
        rgb_nuclei = np.dstack((zero, zero, nuclei_mask.astype(int) * 256, nuclei_mask.astype(float) * 0.5))
        ax.imshow(rgb_nuclei)

        # plot polarity vector
        for index, row in collection.get_properties_by_img_name(img_name).iterrows():
            _add_single_cell_polarity_vector(ax, row["nuc_X"], row["nuc_Y"], row["marker_centroid_X"],
                                             row["marker_centroid_Y"])
            if self.params.show_polarity_angles:
                ax.text(
                    row["nuc_Y"], row["nuc_X"], str(int(np.round(row["marker_nucleus_orientation_deg"], 0))),
                    color="yellow", fontsize=6
                )

        # set title and ax limits
        _add_title(ax, "marker nucleus orientation", im_junction, self.params.show_graphics_axis)

        self._finish_plot(fig, collection.get_out_path_by_name(img_name), img_name, "_marker_nucleus_orientation", [ax],
                          close, nuc_polarity_angle)

    def plot_junction_polarity(self, collection, img_name, close=False):
        im_junction = collection.img_channel_dict[img_name]["junction"]
        cell_mask = collection.masks_dict[img_name].cell_mask_rem_island

        get_logger().info("Plotting: junction polarity")

        # figure and axes
        fig, ax = self._get_figure(1)

        # plot marker intensity
        ax.imshow(im_junction, cmap=plt.cm.gray, alpha=1.0)

        # show cell outlines
        ax.imshow(self._masked_cell_outlines(im_junction, cell_mask), alpha=0.5)

        # add all polarity vectors
        for index, row in collection.get_properties_by_img_name(img_name).iterrows():
            _add_single_cell_polarity_vector(
                ax,
                row["cell_X"],
                row["cell_Y"],
                row["junction_centroid_X"],
                row["junction_centroid_Y"]
            )

        _add_title(ax, "junction polarity", im_junction, self.params.show_graphics_axis)

        self._finish_plot(fig, collection.get_out_path_by_name(img_name), img_name, "_junction_polarity", [ax], close)

    def plot_corners(self, collection, img_name, close=False):
        fig, ax = self._get_figure(1)

        # plot marker intensity
        im_junction = collection.img_channel_dict[img_name]["junction"]
        cell_mask = collection.masks_dict[img_name].cell_mask_rem_island

        ax.imshow(im_junction, cmap=plt.cm.gray, alpha=1.0)

        for index, row in collection.dataset.loc[collection.dataset["filename"] == img_name].iterrows():
            plt.scatter(np.array(json.loads(row["cell_corner_points"]))[:, 0],
                        np.array(json.loads(row["cell_corner_points"]))[:, 1],
                        [4] * len(np.array(json.loads(row["cell_corner_points"]))[:, 1]))

        ax.imshow(self._masked_cell_outlines(im_junction, cell_mask), alpha=0.5)

        _add_title(ax, "cell corners", im_junction, self.params.show_graphics_axis)

        self._finish_plot(fig, collection.get_out_path_by_name(img_name), img_name, "_cell_corners", [ax], close)

    def plot_eccentricity(self, collection, img_name, close=False):
        im_junction = collection.img_channel_dict[img_name]["junction"]
        cell_mask = collection.masks_dict[img_name].cell_mask_rem_island
        nuclei_mask = collection.masks_dict[img_name].nuclei_mask

        get_logger().info("Plotting: eccentricity")

        # figure and axes
        number_sub_figs = 1
        if nuclei_mask is not None:
            nuclei_mask = nuclei_mask.astype(bool)
            number_sub_figs = 2

        fig, ax = self._get_figure(number_sub_figs)

        # get cell_eccentricity
        cell_eccentricity = self._get_polarity_angle_mask(cell_mask, collection, img_name, "cell_eccentricity")

        # add cell (and nuclei) eccentricity to the figure
        if nuclei_mask is not None:
            _add_cell_eccentricity(fig, ax[0], im_junction, cell_mask, cell_eccentricity)
            # get nuclei eccentricity
            nuclei_eccentricity = _calc_nuc_eccentricity(collection.get_properties_by_img_name(img_name), cell_mask,
                                                         nuclei_mask)
            _add_nuclei_eccentricity(fig, ax[1], im_junction, nuclei_mask, nuclei_eccentricity)
        else:
            _add_cell_eccentricity(fig, ax, im_junction, cell_mask, cell_eccentricity)

        # plot major and minor axis
        for index, row in collection.get_properties_by_img_name(img_name).iterrows():
            if nuclei_mask is not None:
                # plot orientation degree
                _add_single_cell_eccentricity_axis(
                    ax[0],
                    row['cell_Y'],
                    row['cell_X'],
                    row['cell_shape_orientation_rad'],
                    row['cell_major_axis_length'],
                    row['cell_minor_axis_length'],
                    row["cell_eccentricity"]
                )

                # plot orientation degree nucleus
                _add_single_cell_eccentricity_axis(
                    ax[1],
                    row['nuc_Y'],
                    row['nuc_X'],
                    row['nuc_shape_orientation_rad'],
                    row['nuc_major_axis_length'],
                    row['nuc_minor_axis_length'],
                    row["nuc_eccentricity"]
                )
            else:
                _add_single_cell_eccentricity_axis(
                    ax,
                    row['cell_Y'],
                    row['cell_X'],
                    row['cell_shape_orientation_rad'],
                    row['cell_major_axis_length'],
                    row['cell_minor_axis_length'],
                    row["cell_eccentricity"]
                )

        # set title and ax limits
        if nuclei_mask is not None:
            _add_title(ax[0], "cell elongation", im_junction, self.params.show_graphics_axis)
            _add_title(ax[1], "nuclei elongation", im_junction, self.params.show_graphics_axis)
            axes = [ax[0], ax[1]]
        else:
            _add_title(ax, "cell elongation", im_junction, self.params.show_graphics_axis)
            axes = [ax]

        self._finish_plot(fig, collection.get_out_path_by_name(img_name), img_name, "_eccentricity", axes, close)

    def plot_ratio_method(self, collection, img_name, close=False):
        im_junction = collection.img_channel_dict[img_name]["junction"]
        cell_mask = collection.masks_dict[img_name].cell_mask_rem_island

        get_logger().info("Plotting: ratio method")

        # figure and axes
        fig, ax = self._get_figure(1)

        # show junction and cell mask overlay
        ax.imshow(im_junction, cmap=plt.cm.gray, alpha=1.0)
        ax.imshow(cell_mask, cmap=plt.cm.Set3, alpha=0.25)

        # show cell outlines
        ax.imshow(self._masked_cell_outlines(im_junction, cell_mask), alpha=0.5)

        # plot major axis around coordinates of each cell
        for index, row in collection.get_properties_by_img_name(img_name).iterrows():
            x0 = row['cell_X']
            y0 = row['cell_Y']

            # upper
            x1 = x0 + math.sin(np.pi / 4.0) * 0.5 * row['cell_major_axis_length']
            y1 = y0 + math.cos(np.pi / 4.0) * 0.5 * row['cell_major_axis_length']
            x2 = x0 + math.cos(np.pi / 4.0) * 0.5 * row['cell_major_axis_length']
            y2 = y0 - math.sin(np.pi / 4.0) * 0.5 * row['cell_major_axis_length']

            ax.plot((y0, y1), (x0, x1), '--r', linewidth=0.5)
            ax.plot((y0, y2), (x0, x2), '--r', linewidth=0.5)
            ax.plot(y0, x0, '.b', markersize=5)

            # lower
            x1 = x0 - math.sin(np.pi / 4.0) * 0.5 * row['cell_major_axis_length']
            y1 = y0 - math.cos(np.pi / 4.0) * 0.5 * row['cell_major_axis_length']
            x2 = x0 - math.cos(np.pi / 4.0) * 0.5 * row['cell_major_axis_length']
            y2 = y0 + math.sin(np.pi / 4.0) * 0.5 * row['cell_major_axis_length']

            ax.plot((y0, y1), (x0, x1), '--r', linewidth=0.5)
            ax.plot((y0, y2), (x0, x2), '--r', linewidth=0.5)
            ax.plot(y0, x0, '.b', markersize=5)

        _add_title(ax, "ratio method", im_junction, self.params.show_graphics_axis)

        self._finish_plot(fig, collection.get_out_path_by_name(img_name), img_name, "_ratio_method", [ax], close)

    def plot_foi(self, collection, img_name, close=False):
        """
        Plot the figure of interest
        """
        get_logger().info("Plotting: figure of interest")

        im_junction = collection.img_channel_dict[img_name]["junction"]
        single_cell_dataset = collection.dataset.loc[collection.dataset["filename"] == img_name]
        foi_name = collection.get_foi_by_img_name(img_name)
        foi = single_cell_dataset[foi_name]
        # figure and axes
        fig, ax = self._get_figure(1)

        cax = ax.imshow(im_junction, cmap=plt.cm.bwr, alpha=1.0)

        # plot the figure of interest
        for index, row in single_cell_dataset.iterrows():
            ax.text(
                row["cell_Y"],
                row["cell_X"],
                str(np.round(row[foi_name], 1)),
                color="w",
                fontsize=5
            )

        yticks = [np.nanmin(foi), np.nanmax(foi, )]
        _add_colorbar(fig, cax, ax, yticks, "intensity cell")

        # set title and ax limits
        _add_title(ax, "feature of interest", im_junction, self.params.show_graphics_axis)

        self._finish_plot(fig, collection.get_out_path_by_name(img_name), img_name, "_foi", [ax], close)

    def plot_orientation(self, collection, img_name, close=False):
        im_junction = collection.img_channel_dict[img_name]["junction"]
        cell_mask = collection.masks_dict[img_name].cell_mask_rem_island
        nuclei_mask = collection.masks_dict[img_name].nuclei_mask

        get_logger().info("Plotting: orientation")

        # figure and axes
        number_sub_figs = 1
        if nuclei_mask is not None:
            nuclei_mask = nuclei_mask.astype(bool)
            number_sub_figs = 2

        fig, ax = self._get_figure(number_sub_figs)

        # get cell_orientation
        cell_orientation = self._get_polarity_angle_mask(cell_mask, collection, img_name, "nuc_shape_orientation_deg")
        # cell_orientation = _calc_cell_orientation(collection.get_properties_by_img_name(img_name), cell_mask)

        # add cell (and nuclei) orientation to the figure
        if nuclei_mask is not None:
            _add_cell_orientation(fig, ax[0], im_junction, cell_mask, cell_orientation)
            # get nuclei orientation
            nuclei_orientation = _calc_nuc_orientation(collection.get_properties_by_img_name(img_name), cell_mask,
                                                       nuclei_mask)
            _add_nuclei_orientation(fig, ax[1], im_junction, nuclei_mask, nuclei_orientation)
        else:
            _add_cell_orientation(fig, ax, im_junction, cell_mask, cell_orientation)

        # plot major and minor axis
        for index, row in collection.get_properties_by_img_name(img_name).iterrows():
            if nuclei_mask is not None:
                # plot orientation degree
                _add_single_cell_orientation_degree_axis(
                    ax[0],
                    row['cell_Y'],
                    row['cell_X'],
                    row['cell_shape_orientation_rad'],
                    row['cell_major_axis_length'],
                    row['cell_minor_axis_length']
                )

                # plot orientation degree nucleus
                _add_single_cell_orientation_degree_axis(
                    ax[1],
                    row['nuc_Y'],
                    row['nuc_X'],
                    row['nuc_shape_orientation_rad'],
                    row['nuc_major_axis_length'],
                    row['nuc_minor_axis_length']
                )
            else:
                # plot orientation degree
                _add_single_cell_orientation_degree_axis(
                    ax,
                    row['cell_Y'],
                    row['cell_X'],
                    row['cell_shape_orientation_rad'],
                    row['cell_major_axis_length'],
                    row['cell_minor_axis_length']
                )

        # set title and ax limits
        if nuclei_mask is not None:
            _add_title(ax[0], "cell shape orientation", im_junction, self.params.show_graphics_axis)
            _add_title(ax[1], "nuclei shape orientation", im_junction, self.params.show_graphics_axis)
            axes = [ax[0], ax[1]]
        else:
            _add_title(ax, "cell shape orientation", im_junction, self.params.show_graphics_axis)
            axes = [ax]

        self._finish_plot(fig, collection.get_out_path_by_name(img_name), img_name, "_shape_orientation", axes, close)

    def _finish_plot(self, fig, output_path, img_name, output_suffix, axes, close=False, image=None):
        # plot scale bar for this figure
        if self.params.plot_scalebar:
            for ax in axes:
                _add_scalebar(
                    ax,
                    self.params.length_scalebar_microns,
                    self.params.pixel_to_micron_ratio,
                    int(self.params.length_scalebar_microns / 2)
                )

        # save output & close
        save_current_fig(
            self.params.graphics_output_format,
            output_path,
            img_name,
            output_suffix,
            image=image
        )

        # close figure
        if close:
            plt.close(fig)

    def plot_collection(self, collection: PropertiesCollection, close=False):
        """Plots the properties dataset"""
        get_logger().info("Plotting...")

        for key in collection.img_channel_dict.keys():

            nuclei_mask = collection.masks_dict[key].nuclei_mask
            organelle_mask = collection.masks_dict[key].organelle_mask
            img_marker = collection.img_channel_dict[key]["marker"]
            img_junction = collection.img_channel_dict[key]["junction"]

            if self.params.plot_polarity and nuclei_mask is not None and organelle_mask is not None:
                self.plot_organelle_polarity(collection, key, close)
                if nuclei_mask is not None:
                    self.plot_nuc_displacement_orientation(collection, key, close)

            if self.params.plot_marker and img_marker is not None:
                self.plot_marker_expression(collection, key, close)
                self.plot_marker_polarity(collection, key, close)
                if nuclei_mask is not None:
                    self.plot_marker_nucleus_orientation(collection, key, close)

            if self.params.plot_junctions and img_junction is not None:
                self.plot_junction_polarity(collection, key, close)
                self.plot_corners(collection, key, close)

            if self.params.plot_orientation:
                self.plot_eccentricity(collection, key, close)

            # Note: disabled for now.
            # if self.params.plot_ratio_method:
            #     self.plot_ratio_method(collection, key, close)

            if self.params.plot_cyclic_orientation:
                self.plot_orientation(collection, key, close)

            # Note: disabled for now.
            # if self.params.plot_foi:
            #     self.plot_foi(collection, key, close)
