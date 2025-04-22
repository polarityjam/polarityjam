"""Module for plotting the features."""
import json
import math
import os
from pathlib import Path
from typing import List, Optional, Union

import cmocean as cm
import matplotlib
import numpy as np
import pandas
import scipy.ndimage as ndi
from matplotlib import pyplot as plt
from shapely.affinity import rotate
from shapely.geometry import LineString

from polarityjam.compute.shape import get_divisor_lines
from polarityjam.compute.statistics import compute_polarity_index
from polarityjam.model.collection import PropertiesCollection
from polarityjam.model.image import BioMedicalChannel, BioMedicalImage
from polarityjam.model.masks import BioMedicalInstanceSegmentationMask, BioMedicalMask
from polarityjam.model.parameter import ImageParameter, PlotParameter
from polarityjam.polarityjam_logging import get_logger
from polarityjam.vizualization.plot import (
    add_colorbar,
    add_scalebar,
    add_title,
    add_vector,
    save_current_fig,
)


class Plotter:
    """Plotter class."""

    def __init__(self, params: PlotParameter):
        """Initialize Plotter."""
        self.params = params
        self.set_figure_dpi()

    def set_figure_dpi(self):
        """Set figure dpi."""
        matplotlib.rcParams["figure.dpi"] = self.params.dpi

    def _get_figure(self, n_subfigures: int):
        w, h = self.params.graphics_width, self.params.graphics_height
        fig, ax = plt.subplots(1, n_subfigures, figsize=(w * n_subfigures, h))
        plt.tight_layout()

        return fig, ax

    def _get_inlines(
        self,
        im_marker: BioMedicalChannel,
        cell_mask: BioMedicalInstanceSegmentationMask,
        nuclei_mask: BioMedicalInstanceSegmentationMask,
        single_cell_dataset: pandas.DataFrame,
    ) -> List[np.ndarray]:
        inlines_cell = BioMedicalInstanceSegmentationMask.empty(im_marker.data.shape)
        inlines_mem = np.zeros((im_marker.data.shape[0], im_marker.data.shape[1]))
        inlines_nuc = BioMedicalInstanceSegmentationMask.empty(im_marker.data.shape)

        for cell_label in single_cell_dataset["label"]:
            feature_row = single_cell_dataset.loc[
                single_cell_dataset["label"] == cell_label
            ]
            intensity_cell = feature_row["marker_mean_expression"].values[0]
            intensity_mem = feature_row["marker_mean_expression_mem"].values[0]

            intensity_nuc = None
            if nuclei_mask is not None:
                intensity_nuc = feature_row["marker_mean_expression_nuc"].values[0]

            single_cell_mask = cell_mask.get_single_instance_mask(cell_label)
            invert_outline_cell = (
                single_cell_mask.get_outline_from_mask(self.params.outline_width)
                .invert()
                .to_instance_mask()
            )
            single_cell_inlay_mask = single_cell_mask.overlay_instance_segmentation(
                invert_outline_cell
            )
            single_cell_inlay_intensity_mask = single_cell_inlay_mask.scalar_mult(
                intensity_cell
            )
            inlines_cell = inlines_cell.element_add(single_cell_inlay_intensity_mask)

            # membrane outline cannot be summed up on an empty mask, because outlines overlap.
            outline_mem = single_cell_mask.get_outline_from_mask(
                self.params.membrane_thickness  # todo: do not use membrane thickness from plot parameters.
            )
            inlines_mem = np.where(
                np.logical_and(outline_mem.data, inlines_mem.data < intensity_mem),
                intensity_mem,
                inlines_mem.data,
            )

            # nuclei marker intensity
            if nuclei_mask is not None:
                single_nucleus_mask = nuclei_mask.get_single_instance_mask(cell_label)
                invert_outline_nuc = (
                    single_nucleus_mask.get_outline_from_mask(self.params.outline_width)
                    .invert()
                    .to_instance_mask()
                )
                single_nuc_inlay_mask = (
                    single_nucleus_mask.overlay_instance_segmentation(
                        invert_outline_nuc
                    )
                )
                single_nuc_inlay_intensity_mask = single_nuc_inlay_mask.scalar_mult(
                    intensity_nuc
                )
                inlines_nuc = inlines_nuc.element_add(single_nuc_inlay_intensity_mask)

        return [inlines_cell.data, inlines_mem, inlines_nuc.data]

    def _get_inlines_junction(
        self,
        img: BioMedicalImage,
        im_junction: BioMedicalChannel,
        cell_mask: BioMedicalInstanceSegmentationMask,
        # collection: PropertiesCollection,
        single_cell_dataset: pandas.DataFrame,
    ) -> List[np.ndarray]:

        inlines_junction_interface_occupancy = np.zeros(
            (im_junction.data.shape[0], im_junction.data.shape[1])
        )
        inlines_junction_intensity_per_interface_area = np.zeros(
            (im_junction.data.shape[0], im_junction.data.shape[1])
        )
        inlines_junction_cluster_density = np.zeros(
            (im_junction.data.shape[0], im_junction.data.shape[1])
        )

        for cell_label in single_cell_dataset["label"]:
            feature_row = single_cell_dataset.loc[
                single_cell_dataset["label"] == cell_label
            ]
            junction_interface_occupancy = feature_row[
                "junction_interface_occupancy"
            ].values[0]
            junction_intensity_per_interface_area = feature_row[
                "junction_intensity_per_interface_area"
            ].values[0]
            junction_cluster_density = feature_row["junction_cluster_density"].values[0]

            single_cell_mask = cell_mask.get_single_instance_mask(cell_label)

            # junction outline cannot be summed up on an empty mask, because outlines overlap.
            outline_junction = single_cell_mask.get_outline_from_mask(
                self.params.membrane_thickness  # todo: do not use membrane thickness from plot parameters.
            )

            sc_junction_mask = img.get_single_junction_mask(
                cell_label,
                self.params.membrane_thickness,  # todo: do not use membrane thickness from plot parameters.
            )

            # TODO: get mask for fragmented junction area (primary feature)
            inlines_junction_interface_occupancy = np.where(
                np.logical_and(
                    sc_junction_mask.data,
                    inlines_junction_interface_occupancy.data
                    < junction_interface_occupancy,
                ),
                junction_interface_occupancy,
                inlines_junction_interface_occupancy.data,
            )

            inlines_junction_intensity_per_interface_area = np.where(
                np.logical_and(
                    outline_junction.data,
                    inlines_junction_intensity_per_interface_area.data
                    < junction_intensity_per_interface_area,
                ),
                junction_intensity_per_interface_area,
                inlines_junction_intensity_per_interface_area.data,
            )

            # TODO: check why we use this logical and here, inline_junction_cluster_density is always 0
            inlines_junction_cluster_density = np.where(
                np.logical_and(
                    sc_junction_mask.data,
                    inlines_junction_cluster_density.data < junction_cluster_density,
                ),
                junction_cluster_density,
                inlines_junction_cluster_density.data,
            )

        # TODO: incorporate all three features
        # junction_interface_occupancy
        # junction_intensity_per_interface_area
        # junction_cluster_density

        return [
            inlines_junction_interface_occupancy,
            inlines_junction_intensity_per_interface_area,
            inlines_junction_cluster_density,
        ]

    def _masked_cell_outlines(
        self,
        channel: BioMedicalChannel,
        instance_seg_mask: BioMedicalInstanceSegmentationMask,
    ) -> np.ndarray:
        # cell outlines
        outlines_cells = BioMedicalMask.empty(channel.data.shape)
        for cell_label in instance_seg_mask.get_labels():
            single_cell_mask = instance_seg_mask.get_single_instance_mask(cell_label)
            outline_cell = single_cell_mask.get_outline_from_mask(
                self.params.outline_width
            )
            outlines_cells = outlines_cells.operation(outline_cell, np.logical_or)

        outlines_cells_instance = outlines_cells.to_instance_mask(
            instance_label=1
        ).mask_background()
        # convert cell outlines to image
        # outlines_cells_rgba = outlines_cells.to_instance_mask()
        # outlines_cells_rgba_stack = np.dstack([outlines_cells_rgba.data] * 3)

        # mask background
        # outlines_cells_rgba_stack = np.ma.masked_where(
        #    outlines_cells_rgba_stack == outlines_cells_rgba.background_label, outlines_cells_rgba_stack
        # )

        return outlines_cells_instance.data

    def plot_channels(
        self,
        seg_img: np.ndarray,
        seg_img_params: ImageParameter,
        output_path: Union[str, Path],
        filename: Union[str, Path],
        close=False,
        cmap="Greys_r",
    ):
        """Plot the separate channels from the input image given, based on its parameters.

        Args:
            seg_img:
                numpy array of the image to plot
            seg_img_params:
                parameters of the image to plot
            output_path:
                path to the output directory where plots are saved
            filename:
                name of the file to save
            close:
                whether to close the figure after saving
            cmap:
                colormap to use for plotting. Default is "Greys_r"

        """
        get_logger().info("Plotting: input channels")

        filename, _ = os.path.splitext(os.path.basename(filename))

        # swap channels if channel last
        if len(seg_img) > 2:
            if seg_img.shape[0] > seg_img.shape[-1]:
                seg_img = np.einsum("ijk->kij", seg_img)

        channel_names, channels = self._get_available_channels(seg_img_params)

        fig, ax = self._get_figure(len(channels))

        if len(channels) == 1:
            # first channel
            ax.imshow(seg_img[:, :], cmap=cmap)
            add_title(
                ax, channel_names[0], seg_img[:, :], self.params.show_graphics_axis
            )
            axes = [ax]
        else:
            for i, c in enumerate(channels):
                ax[i].imshow(seg_img[c, :, :], cmap=cmap)
                add_title(
                    ax[i],
                    channel_names[i],
                    seg_img[c, :, :],
                    self.params.show_graphics_axis,
                )
            axes = ax

        self._finish_plot(
            fig,
            output_path,
            filename,
            "_channels",
            axes,
            seg_img_params.pixel_to_micron_ratio,
            close,
        )

        return fig, axes

    def _get_available_channels(self, seg_img_params):
        channels = []
        channel_names = []
        if (
            seg_img_params.channel_junction is not None
            and seg_img_params.channel_junction >= 0
        ):
            channels.append(seg_img_params.channel_junction)
            channel_names.append("junction channel")
        if (
            seg_img_params.channel_nucleus is not None
            and seg_img_params.channel_nucleus >= 0
        ):
            channels.append(seg_img_params.channel_nucleus)
            channel_names.append("nucleus channel")
        if (
            seg_img_params.channel_organelle is not None
            and seg_img_params.channel_organelle >= 0
        ):
            channels.append(seg_img_params.channel_organelle)
            channel_names.append("organelle channel")
        if (
            seg_img_params.channel_expression_marker is not None
            and seg_img_params.channel_expression_marker >= 0
        ):
            channels.append(seg_img_params.channel_expression_marker)
            channel_names.append("expression marker channel")
        return channel_names, channels

    def plot_mask(
        self,
        mask: np.ndarray,
        seg_img: np.ndarray,
        seg_img_params: ImageParameter,
        output_path: Union[str, Path],
        filename: Union[str, Path],
        mask_nuclei: Optional[np.ndarray] = None,
        mask_golgi: Optional[np.ndarray] = None,
        close: bool = False,
        cmap="Greys_r",
    ):
        """Plot the segmentation mask, together with the separate channels from the input image.

        Args:
            mask:
                numpy array of the mask to plot
            seg_img:
                numpy array of the image to plot
            seg_img_params:
                parameters of the image to plot
            output_path:
                path to the output directory where plots are saved
            filename:
                name of the file to save
            mask_nuclei:
                numpy array of the nuclei mask to plot. If None, no nuclei mask is plotted.
            mask_golgi:
                numpy array of the golgi mask to plot. If None, no golgi mask is plotted.
            close:
                whether to close the figure after saving
            cmap:
                colormap to use for plotting. Default is "Greys_r"

        """
        get_logger().info("Plotting: segmentation masks")

        filename, _ = os.path.splitext(os.path.basename(filename))

        mask_ = self._rand_labels(mask)

        # ignore background
        mask_ = np.where(mask > 0, mask_, np.nan)
        mask_nuclei_ = None

        if mask_nuclei is not None:
            mask_nuclei_ = self._rand_labels(mask_nuclei)

            # ignore background
            mask_nuclei_ = np.where(mask_nuclei > 0, mask_nuclei_, np.nan)

        # swap channels if channel last
        if len(seg_img.shape) > 2:
            if seg_img.shape[0] > seg_img.shape[-1]:
                seg_img = np.einsum("ijc->cij", seg_img)

        channel_names, channels = self._get_available_channels(seg_img_params)

        num_fig = len(channels)
        num_fig = num_fig + 2 if mask_nuclei is not None else num_fig + 1

        mask_golgi_ = None
        if mask_golgi is not None:
            num_fig = num_fig + 1

            mask_golgi_ = self._rand_labels(mask_golgi)

            # ignore background
            mask_golgi_ = np.where(mask_golgi_ > 0, mask_golgi_, np.nan)

        indx_nuc = num_fig - 1

        fig, ax = self._get_figure(num_fig)

        # show channels
        for i, c in enumerate(channels):
            # expand dim
            seg_img = (
                np.expand_dims(seg_img, axis=0) if len(seg_img.shape) == 2 else seg_img
            )
            ax[i].imshow(seg_img[c, :, :], cmap=cmap)
            add_title(
                ax[i],
                channel_names[i],
                seg_img[c, :, :],
                self.params.show_graphics_axis,
            )

        # show mask
        ax[len(channels)].imshow(seg_img[seg_img_params.channel_junction, :, :])
        ax[len(channels)].imshow(
            mask_, cmap=plt.cm.gist_rainbow, alpha=self.params.alpha
        )
        add_title(
            ax[len(channels)],
            "segmentation",
            seg_img[seg_img_params.channel_junction, :, :],
            self.params.show_graphics_axis,
        )

        # show nuclei mask
        if mask_nuclei is not None and seg_img_params.channel_nucleus != -1:
            ax[indx_nuc].imshow(
                seg_img[seg_img_params.channel_nucleus, :, :], cmap=cmap
            )
            ax[indx_nuc].imshow(
                mask_nuclei_, cmap=plt.cm.gist_rainbow, alpha=self.params.alpha
            )
            add_title(
                ax[indx_nuc],
                "segmentation nuclei",
                seg_img[seg_img_params.channel_nucleus, :, :],
                self.params.show_graphics_axis,
            )

        # show golgi mask
        if mask_golgi is not None and seg_img_params.channel_organelle != -1:
            ax[-1].imshow(seg_img[seg_img_params.channel_organelle, :, :], cmap=cmap)
            ax[-1].imshow(
                mask_golgi_, cmap=plt.cm.gist_rainbow, alpha=self.params.alpha
            )
            add_title(
                ax[-1],
                "segmentation golgi",
                seg_img[seg_img_params.channel_organelle, :, :],
                self.params.show_graphics_axis,
            )

        self._finish_plot(
            fig,
            output_path,
            filename,
            "_segmentation",
            ax,
            seg_img_params.pixel_to_micron_ratio,
            close,
        )

        return fig, ax

    def plot_organelle_polarity(
        self, collection: PropertiesCollection, img_name: str, close: bool = False
    ):
        """Plot the organelle polarity of a specific image in the collection.

        Args:
            collection:
                The collection containing the features
            img_name:
                The name of the image to plot
            close:
                whether to close the figure after saving

        """
        img = collection.get_image_by_img_name(img_name)
        assert img.segmentation is not None, "Segmentation is not available"
        assert img.junction is not None, "Junctions channel not available"
        assert img.nucleus is not None, "Nuclei channel not available"
        assert collection.dataset.empty is False, "No data available"
        assert (
            img_name in pandas.Series(list(collection.dataset["filename"])).unique()
        ), "There seems to be no data for the image you selected"
        # assert feature
        assert "organelle_orientation_deg" in list(collection.dataset.columns), (
            "Feature %s not available." % "organelle_orientation_deg"
        )

        im_junction = img.junction.data
        con_inst_seg_mask = img.segmentation.segmentation_mask_connected
        inst_nuclei_mask = img.segmentation.segmentation_mask_nuclei
        inst_organelle_mask = img.segmentation.segmentation_mask_organelle

        pixel_to_micron_ratio = img.img_params.pixel_to_micron_ratio
        r_params = collection.get_runtime_params_by_img_name(img_name)
        cue_direction = r_params.cue_direction

        get_logger().info("Plotting: organelle polarity")

        fig, ax = self._get_figure(1)

        # resources image
        ax.imshow(im_junction, cmap=plt.cm.gray, alpha=1.0)

        # determine polarity_angle
        polarity_angle_vec = collection.get_properties_by_img_name(img_name)[
            "organelle_orientation_deg"
        ].values
        polarity_angle = con_inst_seg_mask.relabel(polarity_angle_vec)

        # plot polarity angle
        cax = ax.imshow(
            polarity_angle.mask_background().data,
            cmap=cm.cm.phase,
            vmin=0,
            vmax=360,
            alpha=self.params.alpha,
        )
        color_bar = fig.colorbar(cax, ax=ax, shrink=0.3)  # , extend='both')
        color_bar.set_label("polarity angle")
        color_bar.ax.set_yticks([0, 90, 180, 270, 360])

        # plot differently colored organelle (red) and nuclei (blue)
        zero = np.zeros((im_junction.shape[0], im_junction.shape[1]))
        rgba_organelle = np.dstack(
            (
                inst_organelle_mask.to_semantic_mask().data,
                zero,
                zero,
                inst_organelle_mask.to_semantic_mask().data,
            )
        )
        rgba_nuclei = np.dstack(
            (
                zero,
                zero,
                inst_nuclei_mask.to_semantic_mask().data,
                inst_nuclei_mask.to_semantic_mask().data,
            )
        )
        ax.imshow(rgba_nuclei)
        ax.imshow(rgba_organelle)

        # plot polarity vector
        for _, row in collection.get_properties_by_img_name(img_name).iterrows():
            add_vector(
                ax,
                row["nuc_X"],
                row["nuc_Y"],
                row["organelle_X"],
                row["organelle_Y"],
                self.params.marker_size,
                self.params.font_color,
            )
            if self.params.show_polarity_angles:
                ax.text(
                    row["cell_X"],
                    row["cell_Y"],
                    str(int(np.round(row["organelle_orientation_deg"], 0))),
                    color=self.params.font_color,
                    fontsize=self.params.fontsize_text_annotations,
                )

        plot_title = "organelle polarity"
        if self.params.show_statistics:
            angles = np.array(
                collection.get_properties_by_img_name(img_name)[
                    "organelle_orientation_rad"
                ]
            )
            cue_direction_rad = np.deg2rad(cue_direction)
            alpha_m, R, c = compute_polarity_index(
                angles, cue_direction=cue_direction_rad, stats_mode="directional"
            )
            plot_title += "\n N: " + str(len(angles)) + ", "
            plot_title += "mean angle: " + str(np.round(alpha_m, 2)) + "°, "
            plot_title += "PI: " + str(np.round(R, 2)) + ","
            plot_title += "\n polarity cue: " + str(np.round(cue_direction, 2)) + "°, "
            plot_title += "c: " + str(np.round(c, 2)) + ", "
            plot_title += "V: " + str(np.round(R * c, 2))

        # show cell outlines
        ax.imshow(
            self._masked_cell_outlines(img.junction, con_inst_seg_mask),
            alpha=self.params.alpha_cell_outline,
            cmap="gray_r",
        )

        # set title and ax limits
        add_title(
            ax,
            plot_title,
            im_junction,
            self.params.show_graphics_axis,
        )

        self._finish_plot(
            fig,
            collection.get_out_path_by_name(img_name),
            img_name,
            "_nuclei_organelle_vector",
            [ax],
            pixel_to_micron_ratio,
            close,
            polarity_angle,
        )

        return fig, [ax]

    def plot_nuc_displacement_orientation(
        self, collection: PropertiesCollection, img_name: str, close: bool = False
    ):
        """Plot the nucleus displacement orientation of a specific image in the collection.

        Args:
            collection:
                The collection containing the features
            img_name:
                The name of the image to plot
            close:
                whether to close the figure after saving

        """
        img = collection.get_image_by_img_name(img_name)
        assert img.segmentation is not None, "Segmentation is not available"
        assert img.junction is not None, "Junctions channel not available"
        assert img.nucleus is not None, "Nuclei channel not available"
        assert collection.dataset.empty is False, "No data available"
        assert (
            img_name in pandas.Series(list(collection.dataset["filename"])).unique()
        ), "There seems to be no data for the image you selected"
        # assert feature
        assert "nuc_displacement_orientation_deg" in list(collection.dataset.columns), (
            "Feature %s not available." % "nuc_displacement_orientation_deg"
        )

        pixel_to_micron_ratio = img.img_params.pixel_to_micron_ratio
        r_params = collection.get_runtime_params_by_img_name(img_name)
        cue_direction = r_params.cue_direction

        get_logger().info("Plotting: marker nucleus polarity")

        # figure and axes
        fig, ax = self._get_figure(1)

        # resources image
        ax.imshow(img.junction.data, cmap=plt.cm.gray, alpha=1.0)

        # determine nucleus polarity_angle
        nuc_polarity_angle_vec = collection.get_properties_by_img_name(img_name)[
            "nuc_displacement_orientation_deg"
        ].values
        nuc_polarity_angle = img.segmentation.segmentation_mask_connected.relabel(
            nuc_polarity_angle_vec
        )

        # plot polarity angle
        cax = ax.imshow(
            nuc_polarity_angle.mask_background().data,
            cmap=cm.cm.phase,
            vmin=0,
            vmax=360,
            alpha=self.params.alpha,
        )
        color_bar = fig.colorbar(cax, ax=ax, shrink=0.3)  # , extend='both')
        color_bar.set_label("polarity angle")
        color_bar.ax.set_yticks([0, 90, 180, 270, 360])

        nuclei_mask = img.segmentation.segmentation_mask_nuclei.to_semantic_mask()

        # plot nuclei (blue)
        zero = np.zeros((img.junction.data.shape[0], img.junction.data.shape[1]))
        rgba_nuclei = np.dstack((zero, zero, nuclei_mask.data, nuclei_mask.data))
        ax.imshow(rgba_nuclei)

        # plot polarity vector
        for _, row in collection.get_properties_by_img_name(img_name).iterrows():
            add_vector(
                ax,
                row["cell_X"],
                row["cell_Y"],
                row["nuc_X"],
                row["nuc_Y"],
                self.params.marker_size,
                self.params.font_color,
            )
            if self.params.show_polarity_angles:
                ax.text(
                    row["nuc_X"],
                    row["nuc_Y"],
                    str(int(np.round(row["nuc_displacement_orientation_deg"], 0))),
                    color=self.params.font_color,
                    fontsize=self.params.fontsize_text_annotations,
                )

        plot_title = "nucleus displacement orientation"

        if self.params.show_statistics:
            angles = np.array(
                collection.get_properties_by_img_name(img_name)[
                    "nuc_displacement_orientation_rad"
                ]
            )
            cue_direction_rad = np.deg2rad(cue_direction)
            alpha_m, R, c = compute_polarity_index(
                angles, cue_direction=cue_direction_rad, stats_mode="directional"
            )
            plot_title += "\n N: " + str(len(angles)) + ", "
            plot_title += "mean angle: " + str(np.round(alpha_m, 2)) + "°, "
            plot_title += "PI: " + str(np.round(R, 2)) + ","
            plot_title += "\n polarity cue: " + str(np.round(cue_direction, 2)) + "°, "
            plot_title += "c: " + str(np.round(c, 2)) + ", "
            plot_title += "V: " + str(np.round(R * c, 2))

        # show cell outlines
        ax.imshow(
            self._masked_cell_outlines(
                img.junction, img.segmentation.segmentation_mask_connected
            ),
            alpha=self.params.alpha_cell_outline,
            cmap="gray_r",
        )

        # set title and ax limits
        add_title(
            ax,
            plot_title,
            img.junction.data,
            self.params.show_graphics_axis,
        )

        self._finish_plot(
            fig,
            collection.get_out_path_by_name(img_name),
            img_name,
            "_nucleus_displacement_orientation",
            [ax],
            pixel_to_micron_ratio,
            close,
            nuc_polarity_angle,
        )

        return fig, [ax]

    def plot_marker_expression(
        self, collection: PropertiesCollection, img_name: str, close: bool = False
    ):
        """Plot the marker expression of a specific image in the collection.

        Args:
            collection:
                The collection containing the features
            img_name:
                The name of the image to plot
            close:
                whether to close the figure after saving

        """
        img = collection.get_image_by_img_name(img_name)
        assert img.segmentation is not None, "Segmentation is not available"
        assert img.marker is not None, "Marker channel not available"
        assert img.junction is not None, "Junctions channel not available"
        assert collection.dataset.empty is False, "No data available"
        assert (
            img_name in pandas.Series(list(collection.dataset["filename"])).unique()
        ), "There seems to be no data for the image you selected"
        # assert feature
        assert "marker_mean_expression" in list(collection.dataset.columns), (
            "Feature %s not available." % "marker_mean_expression"
        )

        im_marker = img.marker
        cell_mask = img.segmentation.segmentation_mask_connected
        single_cell_dataset = collection.dataset.loc[
            collection.dataset["filename"] == img_name
        ]

        nuclei_mask = None
        if img.has_nuclei():
            nuclei_mask = img.segmentation.segmentation_mask_nuclei

        pixel_to_micron_ratio = img.img_params.pixel_to_micron_ratio

        get_logger().info("Plotting: marker expression")
        # figure and axes
        number_sub_figs = 2  # mean intensity cell, mean intensity membrane
        if nuclei_mask is not None:
            number_sub_figs = 3  # (optional) mean intensity nucleus

        fig, ax = self._get_figure(number_sub_figs)

        # plot marker intensity for all subplots
        for i in range(number_sub_figs):
            ax[i].imshow(im_marker.data, cmap=plt.cm.gray, alpha=1.0)

        outlines_cell, outlines_mem, outlines_nuc = self._get_inlines(
            im_marker, cell_mask, nuclei_mask, single_cell_dataset
        )

        # cell and membrane outline
        outlines_cell_ = np.where(outlines_cell > 0, outlines_cell, np.nan)
        cax_1 = ax[0].imshow(outlines_cell_, plt.cm.bwr, alpha=self.params.alpha)

        outlines_mem_ = np.where(outlines_mem > 0, outlines_mem, np.nan)
        cax_2 = ax[1].imshow(outlines_mem_, plt.cm.bwr, alpha=self.params.alpha)

        # nuclei marker intensity
        cax_3 = None
        outlines_nuc_ = None
        if nuclei_mask is not None:
            outlines_nuc_ = np.where(outlines_nuc > 0, outlines_nuc, np.nan)
            cax_3 = ax[2].imshow(
                outlines_nuc_, plt.cm.bwr, alpha=self.params.alpha
            )  # always last axis

        # colorbar for cell
        yticks_cell = [
            np.nanmin(outlines_cell_),
            np.nanmax(
                outlines_cell_,
            ),
        ]
        add_colorbar(fig, cax_1, ax[0], yticks_cell, "intensity cell")

        # colorbar for membrane
        yticks_mem = [
            np.nanmin(outlines_mem_),
            np.nanmax(
                outlines_mem_,
            ),
        ]
        add_colorbar(fig, cax_2, ax[1], yticks_mem, "intensity membrane")

        # colorbar for nucleus
        if nuclei_mask is not None:
            yticks_nuc = [np.nanmin(outlines_nuc_), np.nanmax(outlines_nuc_)]
            add_colorbar(fig, cax_3, ax[2], yticks_nuc, "intensity nucleus")

        # plot mean expression value of cell and membrane as text
        for _, row in single_cell_dataset.iterrows():
            ax[0].text(
                row["cell_X"],
                row["cell_Y"],
                str(np.round(row["marker_mean_expression"], 1)),
                color=self.params.font_color,
                fontsize=self.params.fontsize_text_annotations,
            )
            ax[1].text(
                row["cell_X"],
                row["cell_Y"],
                str(np.round(row["marker_mean_expression_mem"], 1)),
                color=self.params.font_color,
                fontsize=self.params.fontsize_text_annotations,
            )
            if nuclei_mask is not None:
                ax[2].text(
                    row["nuc_X"],
                    row["nuc_Y"],
                    str(np.round(row["marker_mean_expression_nuc"], 1)),
                    color=self.params.font_color,
                    fontsize=self.params.fontsize_text_annotations,
                )

        # set title
        axes = [ax[0], ax[1]]
        add_title(
            ax[0],
            "marker mean intensity cell",
            im_marker.data,
            self.params.show_graphics_axis,
        )
        add_title(
            ax[1],
            "marker mean intensity membrane",
            im_marker.data,
            self.params.show_graphics_axis,
        )
        if nuclei_mask is not None:
            add_title(
                ax[2],
                "marker mean intensity nucleus",
                im_marker.data,
                self.params.show_graphics_axis,
            )
            axes = [ax[0], ax[1], ax[2]]

        # show cell outlines
        ax[0].imshow(
            self._masked_cell_outlines(img.junction, cell_mask),
            alpha=self.params.alpha_cell_outline,
            cmap="gray_r",
        )
        if nuclei_mask is not None:
            # show cell outlines
            ax[2].imshow(
                self._masked_cell_outlines(img.junction, cell_mask),
                alpha=self.params.alpha_cell_outline,
                cmap="gray_r",
            )

        self._finish_plot(
            fig,
            collection.get_out_path_by_name(img_name),
            img_name,
            "_marker_expression",
            axes,
            pixel_to_micron_ratio,
            close,
        )

        return fig, axes

    def plot_marker_polarity(
        self, collection: PropertiesCollection, img_name: str, close: bool = False
    ):
        """Plot the marker polarity of a specific image in the collection.

        Args:
            collection:
                The collection containing the features
            img_name:
                The name of the image to plot
            close:
                whether to close the figure after saving

        """
        img = collection.get_image_by_img_name(img_name)
        assert img.segmentation is not None, "Segmentation is not available"
        assert img.marker is not None, "Marker channel not available"
        assert collection.dataset.empty is False, "No data available"
        assert (
            img_name in pandas.Series(list(collection.dataset["filename"])).unique()
        ), "There seems to be no data for the image you selected"
        # assert feature
        assert "marker_centroid_orientation_deg" in list(collection.dataset.columns), (
            "Feature %s not available." % "marker_centroid_orientation_deg"
        )

        im_marker = img.marker
        cell_mask = img.segmentation.segmentation_mask_connected

        pixel_to_micron_ratio = img.img_params.pixel_to_micron_ratio
        r_params = collection.get_runtime_params_by_img_name(img_name)
        cue_direction = r_params.cue_direction

        get_logger().info("Plotting: marker polarity")

        # figure and axes
        fig, ax = self._get_figure(1)

        # plot marker intensity
        im_marker_ = ndi.gaussian_filter(im_marker.data, sigma=1)
        cax = ax.imshow(im_marker_, cmap=plt.cm.gray, alpha=1.0)

        # nanmin = np.nanpercentile(im_marker_,10)
        # nanmax = np.nanpercentile(im_marker_,90)

        # nanmin = np.nanmin(im_marker_)
        # nanmax = np.nanmax(im_marker_)
        # yticks = [nanmin, np.round(nanmin + (nanmax - nanmin) / 2, 1), nanmax]
        # add_colorbar(fig, cax, ax, yticks, "marker intensity")

        # determine marker polarity_angle and show cells in colors
        marker_polarity_orientation_deg_vec = collection.get_properties_by_img_name(
            img_name
        )["marker_centroid_orientation_deg"].values
        marker_polarity_angle = cell_mask.relabel(marker_polarity_orientation_deg_vec)

        # plot polarity angle
        cax = ax.imshow(
            marker_polarity_angle.mask_background().data,
            cmap=cm.cm.phase,
            vmin=0,
            vmax=360,
            alpha=self.params.alpha,
        )
        color_bar = fig.colorbar(cax, ax=ax, shrink=0.3)  # , extend='both')
        color_bar.set_label("polarity angle")
        color_bar.ax.set_yticks([0, 90, 180, 270, 360])

        # add all polarity vectors
        for _, row in collection.get_properties_by_img_name(img_name).iterrows():
            add_vector(
                ax,
                row["cell_X"],
                row["cell_Y"],
                row["marker_centroid_X"],
                row["marker_centroid_Y"],
                self.params.marker_size,
                self.params.font_color,
            )
            if self.params.show_polarity_angles:
                ax.text(
                    row["cell_X"],
                    row["cell_Y"],
                    str(int(np.round(row["marker_centroid_orientation_deg"], 0))),
                    color=self.params.font_color,
                    fontsize=self.params.fontsize_text_annotations,
                )

        plot_title = "marker polarity"

        if self.params.show_statistics:
            angles = np.array(
                collection.get_properties_by_img_name(img_name)[
                    "marker_centroid_orientation_rad"
                ]
            )
            cue_direction_rad = np.deg2rad(cue_direction)
            alpha_m, R, c = compute_polarity_index(
                angles, cue_direction=cue_direction_rad, stats_mode="directional"
            )
            # plot_title += "\n mean \u03B1: " + str(np.round(alpha_m, 2)) + "°, "
            plot_title += "\n N: " + str(len(angles)) + ", "
            plot_title += "mean angle: " + str(np.round(alpha_m, 2)) + "°, "
            plot_title += "PI: " + str(np.round(R, 2)) + ","
            plot_title += "\n polarity cue: " + str(np.round(alpha_m, 2)) + "°, "
            plot_title += "c: " + str(np.round(c, 2)) + ", "
            plot_title += "V: " + str(np.round(R * c, 2))

        # show cell outlines
        ax.imshow(
            self._masked_cell_outlines(im_marker, cell_mask),
            alpha=self.params.alpha_cell_outline,
            cmap="gray_r",
        )

        add_title(ax, plot_title, im_marker.data, self.params.show_graphics_axis)

        self._finish_plot(
            fig,
            collection.get_out_path_by_name(img_name),
            img_name,
            "_marker_polarity",
            [ax],
            pixel_to_micron_ratio,
            close,
        )

        return fig, [ax]

    def plot_marker_nucleus_orientation(
        self, collection: PropertiesCollection, img_name: str, close: bool = False
    ):
        """Plot the marker polarity of a specific image in the collection.

        Args:
            collection:
                the collection containing the features
            img_name:
                the name of the image to plot
            close:
                whether to close the figure after saving

        """
        img = collection.get_image_by_img_name(img_name)
        assert img.segmentation is not None, "Segmentation is not available"
        assert (
            img.segmentation.segmentation_mask_nuclei is not None
        ), "Nuclei segmentation not available"
        assert img.junction is not None, "Junction channel not available"
        assert img.nucleus is not None, "Nuclei channel not available"
        assert img.marker is not None, "Marker channel not available"
        assert collection.dataset.empty is False, "No data available"
        assert (
            img_name in pandas.Series(list(collection.dataset["filename"])).unique()
        ), "There seems to be no data for the image you selected"
        # assert feature
        assert "marker_nucleus_orientation_deg" in list(collection.dataset.columns), (
            "Feature %s not available." % "marker_nucleus_orientation_deg"
        )

        im_junction = img.junction.data
        segmentation_mask = img.segmentation.segmentation_mask_connected
        inst_nuclei_mask = img.segmentation.segmentation_mask_nuclei

        pixel_to_micron_ratio = img.img_params.pixel_to_micron_ratio
        r_params = collection.get_runtime_params_by_img_name(img_name)
        cue_direction = r_params.cue_direction

        get_logger().info("Plotting: marker nucleus polarity")

        # figure and axes
        fig, ax = self._get_figure(1)

        # resources image
        ax.imshow(img.marker.data, cmap=plt.cm.gray, alpha=1.0)

        # determine nucleus polarity_angle
        marker_nucleus_orientation_deg_vec = collection.get_properties_by_img_name(
            img_name
        )["marker_nucleus_orientation_deg"].values
        nuc_polarity_angle = segmentation_mask.relabel(
            marker_nucleus_orientation_deg_vec
        )

        # plot polarity angle
        cax = ax.imshow(
            nuc_polarity_angle.mask_background().data,
            cmap=cm.cm.phase,
            vmin=0,
            vmax=360,
            alpha=self.params.alpha,
        )
        color_bar = fig.colorbar(cax, ax=ax, shrink=0.3)  # , extend='both')
        color_bar.set_label("polarity angle")
        color_bar.ax.set_yticks([0, 90, 180, 270, 360])

        # plot nuclei (blue)
        zero = np.zeros((im_junction.shape[0], im_junction.shape[1]))
        rgba_nuclei = np.dstack(
            (
                zero,
                zero,
                inst_nuclei_mask.to_semantic_mask().data,
                inst_nuclei_mask.to_semantic_mask().data,
            )
        )
        ax.imshow(rgba_nuclei)

        # plot polarity vector
        for _, row in collection.get_properties_by_img_name(img_name).iterrows():
            add_vector(
                ax,
                row["nuc_X"],
                row["nuc_Y"],
                row["marker_centroid_X"],
                row["marker_centroid_Y"],
                self.params.marker_size,
                self.params.font_color,
            )
            if self.params.show_polarity_angles:
                ax.text(
                    row["nuc_X"],
                    row["nuc_Y"],
                    str(int(np.round(row["marker_nucleus_orientation_deg"], 0))),
                    color=self.params.font_color,
                    fontsize=self.params.fontsize_text_annotations,
                )

        plot_title = "marker nucleus orientation"

        if self.params.show_statistics:
            angles = np.array(
                collection.get_properties_by_img_name(img_name)[
                    "marker_nucleus_orientation_rad"
                ]
            )
            cue_direction_rad = np.deg2rad(cue_direction)
            alpha_m, R, c = compute_polarity_index(
                angles, cue_direction=cue_direction_rad, stats_mode="directional"
            )
            # plot_title += "\n mean \u03B1: " + str(np.round(alpha_m, 2)) + "°, "
            plot_title += "\n N: " + str(len(angles)) + ", "
            plot_title += "mean angle: " + str(np.round(alpha_m, 2)) + "°, "
            plot_title += "PI: " + str(np.round(R, 2)) + ","
            plot_title += "\n polarity cue: " + str(np.round(cue_direction, 2)) + "°, "
            plot_title += "c: " + str(np.round(c, 2)) + ", "
            plot_title += "V: " + str(np.round(R * c, 2))

        # show cell outlines
        ax.imshow(
            self._masked_cell_outlines(img.junction, segmentation_mask),
            alpha=self.params.alpha_cell_outline,
            cmap="gray_r",
        )

        # set title and ax limits
        add_title(
            ax,
            plot_title,
            im_junction,
            self.params.show_graphics_axis,
        )

        self._finish_plot(
            fig,
            collection.get_out_path_by_name(img_name),
            img_name,
            "_marker_nucleus_orientation",
            [ax],
            pixel_to_micron_ratio,
            close,
            nuc_polarity_angle,
        )

        return fig, [ax]

    def plot_junction_features(
        self, collection: PropertiesCollection, img_name: str, close: bool = False
    ):
        """Plot the junction features of a specific image in the collection.

        Args:
            collection:
                The collection containing the features
            img_name:
                The name of the image to plot
            close:
                whether to close the figure after saving

        """
        img = collection.get_image_by_img_name(img_name)
        assert img.segmentation is not None, "Segmentation is not available"
        assert img.junction is not None, "Junction channel not available"
        assert collection.dataset.empty is False, "No data available"
        assert (
            img_name in pandas.Series(list(collection.dataset["filename"])).unique()
        ), "There seems to be no data for the image you selected"
        # assert feature
        assert "junction_interface_occupancy" in list(
            collection.dataset.columns
        ), "Junction features %s not available."

        im_junction = img.junction
        cell_mask = img.segmentation.segmentation_mask_connected
        single_cell_dataset = collection.dataset.loc[
            collection.dataset["filename"] == img_name
        ]

        pixel_to_micron_ratio = img.img_params.pixel_to_micron_ratio
        collection.get_runtime_params_by_img_name(img_name)

        get_logger().info("Plotting: junction features")
        number_sub_figs = 4
        # figure and axes
        fig, ax = self._get_figure(number_sub_figs)

        # plot junction channel for all subplots
        cax = []
        for i in range(number_sub_figs):
            _cax = ax[i].imshow(im_junction.data, cmap=plt.cm.gray, alpha=1.0)
            cax.append(_cax)

        # add color bar for plain junction channel
        yticks_mem = [
            np.nanmin(im_junction.data),
            np.nanmax(
                im_junction.data,
            ),
        ]
        add_colorbar(fig, cax[0], ax[0], yticks_mem, "junction intensity")

        add_title(
            ax[0],
            "junction channel",
            im_junction.data,
            self.params.show_graphics_axis,
        )

        (
            outlines_junction_interface_occupancy,
            outlines_junction_intensity_per_interface_area,
            outlines_junction_cluster_density,
        ) = self._get_inlines_junction(img, im_junction, cell_mask, single_cell_dataset)

        outlines_junction_ = np.where(
            outlines_junction_interface_occupancy > 0,
            outlines_junction_interface_occupancy,
            np.nan,
        )
        cax_1 = ax[1].imshow(outlines_junction_, plt.cm.bwr, alpha=self.params.alpha)

        # colorbar for membrane
        yticks_mem = [
            np.nanmin(outlines_junction_),
            np.nanmax(
                outlines_junction_,
            ),
        ]
        add_colorbar(fig, cax_1, ax[1], yticks_mem, "junction interface occupancy")
        add_title(
            ax[1],
            "junction interface occupancy",
            im_junction.data,
            self.params.show_graphics_axis,
        )

        outlines_junction_ = np.where(
            outlines_junction_intensity_per_interface_area > 0,
            outlines_junction_intensity_per_interface_area,
            np.nan,
        )
        cax_2 = ax[2].imshow(outlines_junction_, plt.cm.bwr, alpha=self.params.alpha)

        # colorbar for membrane
        yticks_mem = [
            np.nanmin(outlines_junction_),
            np.nanmax(
                outlines_junction_,
            ),
        ]
        add_colorbar(
            fig, cax_2, ax[2], yticks_mem, "junction intensity per interface area"
        )
        add_title(
            ax[2],
            "junction intensity per interface area",
            im_junction.data,
            self.params.show_graphics_axis,
        )

        outlines_junction_ = np.where(
            outlines_junction_cluster_density > 0,
            outlines_junction_cluster_density,
            np.nan,
        )
        cax_3 = ax[3].imshow(outlines_junction_, plt.cm.bwr, alpha=self.params.alpha)

        # colorbar for membrane
        yticks_mem = [
            np.nanmin(outlines_junction_),
            np.nanmax(
                outlines_junction_,
            ),
        ]
        add_colorbar(fig, cax_3, ax[3], yticks_mem, "junction cluster density")
        add_title(
            ax[3],
            "junction cluster density",
            im_junction.data,
            self.params.show_graphics_axis,
        )

        axes = [ax[0], ax[1], ax[2], ax[3]]

        self._finish_plot(
            fig,
            collection.get_out_path_by_name(img_name),
            img_name,
            "_junction_features",
            axes,
            pixel_to_micron_ratio,
            close,
        )

        return fig, axes

    def plot_junction_polarity(
        self, collection: PropertiesCollection, img_name: str, close: bool = False
    ):
        """Plot the junction polarity of a specific image in the collection.

        Args:
            collection:
                The collection containing the features
            img_name:
                The name of the image to plot
            close:
                whether to close the figure after saving

        """
        img = collection.get_image_by_img_name(img_name)
        assert img.segmentation is not None, "Segmentation is not available"
        assert img.junction is not None, "Junction channel not available"
        assert collection.dataset.empty is False, "No data available"
        assert (
            img_name in pandas.Series(list(collection.dataset["filename"])).unique()
        ), "There seems to be no data for the image you selected"
        # assert feature
        assert "junction_centroid_orientation_deg" in list(
            collection.dataset.columns
        ), ("Features %s not available." % "junction_centroid_orientation_deg")

        im_junction = img.junction
        cell_mask = img.segmentation.segmentation_mask_connected

        pixel_to_micron_ratio = img.img_params.pixel_to_micron_ratio
        collection.get_runtime_params_by_img_name(img_name)

        get_logger().info("Plotting: junction polarity")

        # figure and axes
        fig, ax = self._get_figure(1)

        # plot marker intensity
        ax.imshow(im_junction.data, cmap=plt.cm.gray, alpha=1.0)

        # determine polarity_angle
        polarity_angle_vec = collection.get_properties_by_img_name(img_name)[
            "junction_centroid_orientation_deg"
        ].values
        polarity_angle = cell_mask.relabel(polarity_angle_vec)

        # plot polarity angle
        cax = ax.imshow(
            polarity_angle.mask_background().data,
            cmap=cm.cm.phase,
            vmin=0,
            vmax=360,
            alpha=self.params.alpha,
        )
        color_bar = fig.colorbar(cax, ax=ax, shrink=0.3)  # , extend='both')
        color_bar.set_label("polarity angle")
        color_bar.ax.set_yticks([0, 90, 180, 270, 360])

        # add all polarity vectors
        for _, row in collection.get_properties_by_img_name(img_name).iterrows():
            add_vector(
                ax,
                row["cell_X"],
                row["cell_Y"],
                row["junction_centroid_X"],
                row["junction_centroid_Y"],
                self.params.marker_size,
                self.params.font_color,
            )
            if self.params.show_polarity_angles:
                ax.text(
                    row["cell_X"],
                    row["cell_Y"],
                    str(int(np.round(row["junction_centroid_orientation_deg"], 0))),
                    color=self.params.font_color,
                    fontsize=self.params.fontsize_text_annotations,
                )

        plot_title = "junction polarity"
        # if self.params.plot_statistics:
        #    angles = np.array(collection.get_properties_by_img_name(img_name)["junction_polarity_rad"])
        #    alpha_m, R, c = compute_polarity_index(
        #        angles, cue_direction=r_params.cue_direction, stats_mode='directional'
        #    )
        #    plot_title += "\n mean \u03B1: " + str(np.round(alpha_m, 2)) + "°, "
        #    plot_title += "PI: " + str(np.round(R, 2)) + ","
        #    plot_title += "\n c: " + str(np.round(c, 2))
        #    plot_title += ", V: " + str(np.round(R * c, 2))

        # show cell outlines
        ax.imshow(
            self._masked_cell_outlines(im_junction, cell_mask),
            alpha=self.params.alpha_cell_outline,
            cmap="gray_r",
        )

        add_title(ax, plot_title, im_junction.data, self.params.show_graphics_axis)

        self._finish_plot(
            fig,
            collection.get_out_path_by_name(img_name),
            img_name,
            "_junction_polarity",
            [ax],
            pixel_to_micron_ratio,
            close,
        )

        return fig, [ax]

    def plot_corners(
        self, collection: PropertiesCollection, img_name: str, close: bool = False
    ):
        """Plot the corners of a specific image in the collection.

        Args:
            collection:
                The collection containing the features
            img_name:
                The name of the image to plot
            close:
                whether to close the figure after saving

        """
        fig, ax = self._get_figure(1)

        img = collection.get_image_by_img_name(img_name)
        assert img.segmentation is not None, "Segmentation is not available"
        assert img.junction is not None, "Junction channel not available"
        assert collection.dataset.empty is False, "No data available"
        assert (
            img_name in pandas.Series(list(collection.dataset["filename"])).unique()
        ), "There seems to be no data for the image you selected"
        # assert feature
        assert "cell_corner_points" in list(collection.dataset.columns), (
            "Features %s not available." % "cell_corner_points"
        )

        pixel_to_micron_ratio = img.img_params.pixel_to_micron_ratio

        # plot marker intensity
        im_junction = img.junction
        cell_mask = img.segmentation.segmentation_mask_connected

        ax.imshow(im_junction.data, cmap=plt.cm.gray, alpha=1.0)

        for _, row in collection.dataset.loc[
            collection.dataset["filename"] == img_name
        ].iterrows():
            plt.scatter(
                np.array(json.loads(row["cell_corner_points"]))[:, 0],
                np.array(json.loads(row["cell_corner_points"]))[:, 1],
                [4] * len(np.array(json.loads(row["cell_corner_points"]))[:, 1]),
            )

        ax.imshow(
            self._masked_cell_outlines(im_junction, cell_mask),
            alpha=self.params.alpha_cell_outline,
            cmap="gray_r",
        )

        add_title(ax, "cell corners", im_junction.data, self.params.show_graphics_axis)

        self._finish_plot(
            fig,
            collection.get_out_path_by_name(img_name),
            img_name,
            "_cell_corners",
            [ax],
            pixel_to_micron_ratio,
            close,
        )

        return fig, [ax]

    def plot_eccentricity(
        self, collection: PropertiesCollection, img_name: str, close: bool = False
    ):
        """Plot the eccentricity of a specific image in the collection.

        Args:
            collection:
                The collection containing the features
            img_name:
                The name of the image to plot
            close:
                whether to close the figure after saving

        """
        img = collection.get_image_by_img_name(img_name)
        assert img.segmentation is not None, "Segmentation is not available"
        assert img.junction is not None, "Junction channel not available"
        assert collection.dataset.empty is False, "No data available"
        assert (
            img_name in pandas.Series(list(collection.dataset["filename"])).unique()
        ), "There seems to be no data for the image you selected"
        # assert feature
        assert "cell_eccentricity" in list(collection.dataset.columns), (
            "Features %s not available." % "cell_eccentricity"
        )

        im_junction = img.junction
        segmentation_mask = img.segmentation.segmentation_mask_connected
        inst_nuclei_mask = None
        if img.has_nuclei():
            assert (
                img.segmentation.segmentation_mask_nuclei is not None
            ), "Nuclei segmentation not available"
            inst_nuclei_mask = img.segmentation.segmentation_mask_nuclei

        pixel_to_micron_ratio = img.img_params.pixel_to_micron_ratio

        get_logger().info("Plotting: eccentricity")

        # figure and axes
        number_sub_figs = 1
        if inst_nuclei_mask is not None:
            number_sub_figs = 2

        fig, ax = self._get_figure(number_sub_figs)

        # get cell_eccentricity
        cell_eccentricity_vec = collection.get_properties_by_img_name(img_name)[
            "cell_eccentricity"
        ].values
        cell_eccentricity = segmentation_mask.relabel(cell_eccentricity_vec)

        # add cell (and nuclei) eccentricity to the figure
        if inst_nuclei_mask is not None:
            Plotter._add_values_0_1(
                fig,
                ax[0],
                im_junction,
                cell_eccentricity,
                self.params.alpha,
                "eccentricity",
            )
            # get nuclei eccentricity
            nuclei_eccentricity_vec = collection.get_properties_by_img_name(img_name)[
                "nuc_eccentricity"
            ].values
            nuclei_eccentricity = inst_nuclei_mask.relabel(nuclei_eccentricity_vec)

            get_logger().info(
                "Maximal nuclei eccentricity: %s"
                % str(np.max(nuclei_eccentricity.data))
            )
            get_logger().info(
                "Minimal nuclei eccentricity: %s"
                % str(np.min(nuclei_eccentricity.data))
            )

            Plotter._add_values_0_1(
                fig,
                ax[1],
                im_junction,
                nuclei_eccentricity,
                self.params.alpha,
                "eccentricity",
            )

        else:
            Plotter._add_values_0_1(
                fig,
                ax,
                im_junction,
                cell_eccentricity,
                self.params.alpha,
                "eccentricity",
            )

        # plot major and minor axis
        for _, row in collection.get_properties_by_img_name(img_name).iterrows():
            if inst_nuclei_mask is not None:
                # plot orientation degree
                Plotter._add_single_cell_length_to_width_ratio_axis(
                    ax[0],
                    row["cell_X"],
                    row["cell_Y"],
                    row["cell_shape_orientation_rad"],
                    row["cell_major_axis_length"],
                    row["cell_minor_axis_length"],
                    row["cell_eccentricity"],
                    self.params.fontsize_text_annotations,
                    self.params.font_color,
                    self.params.marker_size,
                )

                # plot orientation degree nucleus
                Plotter._add_single_cell_length_to_width_ratio_axis(
                    ax[1],
                    row["nuc_X"],
                    row["nuc_Y"],
                    row["nuc_shape_orientation_rad"],
                    row["nuc_major_axis_length"],
                    row["nuc_minor_axis_length"],
                    row["nuc_eccentricity"],
                    self.params.fontsize_text_annotations,
                    self.params.font_color,
                    self.params.marker_size,
                )
            else:
                Plotter._add_single_cell_length_to_width_ratio_axis(
                    ax,
                    row["cell_X"],
                    row["cell_Y"],
                    row["cell_shape_orientation_rad"],
                    row["cell_major_axis_length"],
                    row["cell_minor_axis_length"],
                    row["cell_eccentricity"],
                    self.params.fontsize_text_annotations,
                    self.params.font_color,
                    self.params.marker_size,
                )

        if inst_nuclei_mask is not None:
            # show cell outlines
            ax[0].imshow(
                self._masked_cell_outlines(im_junction, segmentation_mask),
                alpha=self.params.alpha_cell_outline,
                cmap="gray_r",
            )
            # show cell outlines
            ax[1].imshow(
                self._masked_cell_outlines(im_junction, segmentation_mask),
                alpha=self.params.alpha_cell_outline,
                cmap="gray_r",
            )
        else:
            # show cell outlines
            ax.imshow(
                self._masked_cell_outlines(im_junction, segmentation_mask),
                alpha=self.params.alpha_cell_outline,
                cmap="gray_r",
            )

        # set title and ax limits
        if inst_nuclei_mask is not None:
            add_title(
                ax[0],
                "cell eccentricity",
                im_junction.data,
                self.params.show_graphics_axis,
            )
            add_title(
                ax[1],
                "nuclei eccentricity",
                im_junction.data,
                self.params.show_graphics_axis,
            )
            axes = [ax[0], ax[1]]
        else:
            add_title(
                ax,
                "cell eccentricity",
                im_junction.data,
                self.params.show_graphics_axis,
            )
            axes = [ax]

        self._finish_plot(
            fig,
            collection.get_out_path_by_name(img_name),
            img_name,
            "_eccentricity",
            axes,
            pixel_to_micron_ratio,
            close,
        )

        return fig, axes

    def plot_length_to_width_ratio(
        self, collection: PropertiesCollection, img_name: str, close: bool = False
    ):
        """Plot the length to width ratio of cells (and nuclei) in a specific image in the collection.

        Args:
            collection:
                The collection containing the features
            img_name:
                The name of the image to plot
            close:
                whether to close the figure after saving

        """
        img = collection.get_image_by_img_name(img_name)
        assert img.segmentation is not None, "Segmentation is not available"
        assert img.junction is not None, "Junction channel not available"
        assert collection.dataset.empty is False, "No data available"
        assert (
            img_name in pandas.Series(list(collection.dataset["filename"])).unique()
        ), "There seems to be no data for the image you selected"
        # assert feature
        assert "cell_length_to_width_ratio" in list(collection.dataset.columns), (
            "Features %s not available." % "cell_length_to_width_ratio"
        )
        assert "cell_shape_orientation_rad" in list(collection.dataset.columns), (
            "Features %s not available."
            % "cell_shape_orientation_rad. Needed for orientation of the major axis."
        )

        im_junction = img.junction
        segmentation_mask = img.segmentation.segmentation_mask_connected

        inst_nuclei_mask = None
        if img.has_nuclei():
            assert (
                img.segmentation.segmentation_mask_nuclei is not None
            ), "Nuclei segmentation not available"
            inst_nuclei_mask = img.segmentation.segmentation_mask_nuclei

        pixel_to_micron_ratio = img.img_params.pixel_to_micron_ratio

        get_logger().info("Plotting: length to width ratio (elongation)")

        # figure and axes
        number_sub_figs = 1
        if inst_nuclei_mask is not None:
            number_sub_figs = 2

        fig, ax = self._get_figure(number_sub_figs)

        # get cell_eccentricity
        single_cell_dataset = collection.dataset.loc[
            collection.dataset["filename"] == img_name
        ]

        # add cell (and nuclei) eccentricity to the figure
        if inst_nuclei_mask is not None:
            ax[0].imshow(im_junction.data, cmap=plt.cm.gray, alpha=1.0)

            LWR_values = single_cell_dataset["cell_length_to_width_ratio"]
            # plot the figure of interest
            m = np.copy(segmentation_mask.data)
            for _, row in single_cell_dataset.iterrows():
                LWR_val = row["cell_length_to_width_ratio"]
                label = row["label"]

                m = np.where(segmentation_mask.data == label, LWR_val, m)

            cax_0 = ax[0].imshow(
                np.ma.masked_where(m == 0, m), cmap=plt.cm.bwr, alpha=self.params.alpha
            )

            nanmin = np.round(np.nanmin(LWR_values), 1)
            nanmax = np.round(np.nanmax(LWR_values), 1)
            yticks = [nanmin, np.round(nanmin + (nanmax - nanmin) / 2, 1), nanmax]
            add_colorbar(fig, cax_0, ax[0], yticks, "length to width ratio")

            ax[1].imshow(im_junction.data, cmap=plt.cm.gray, alpha=1)

            nuc_LWR_values = single_cell_dataset["nuc_length_to_width_ratio"]
            # plot the figure of interest
            m = np.copy(inst_nuclei_mask.data)
            for _, row in single_cell_dataset.iterrows():
                LWR_val = row[("nuc_length_to_width_ratio")]
                label = row["label"]

                m = np.where(inst_nuclei_mask.data == label, LWR_val, m)

            cax_1 = ax[1].imshow(
                np.ma.masked_where(m == 0, m), cmap=plt.cm.bwr, alpha=self.params.alpha
            )

            nanmin = np.round(np.nanmin(nuc_LWR_values), 1)
            nanmax = np.round(np.nanmax(nuc_LWR_values), 1)
            yticks = [nanmin, np.round(nanmin + (nanmax - nanmin) / 2, 1), nanmax]
            add_colorbar(fig, cax_1, ax[1], yticks, "length to width ratio")

            # cell outlines
            ax[0].imshow(
                self._masked_cell_outlines(im_junction, segmentation_mask),
                alpha=self.params.alpha_cell_outline,
                cmap="gray_r",
            )

            # cell outlines
            ax[1].imshow(
                self._masked_cell_outlines(im_junction, segmentation_mask),
                alpha=self.params.alpha_cell_outline,
                cmap="gray_r",
            )

        else:
            ax.imshow(im_junction.data, cmap=plt.cm.gray, alpha=1.0)

            LWR_values = single_cell_dataset["cell_length_to_width_ratio"]
            # plot the figure of interest
            m = np.copy(segmentation_mask.data)
            for _, row in single_cell_dataset.iterrows():
                LWR_val = row["cell_length_to_width_ratio"]
                label = row["label"]

                m = np.where(segmentation_mask.data == label, LWR_val, m)

            cax = ax.imshow(
                np.ma.masked_where(m == 0, m), cmap=plt.cm.bwr, alpha=self.params.alpha
            )

            nanmin = np.round(np.nanmin(LWR_values), 1)
            nanmax = np.round(np.nanmax(LWR_values), 1)
            yticks = [nanmin, np.round(nanmin + (nanmax - nanmin) / 2, 1), nanmax]
            add_colorbar(fig, cax, ax, yticks, "length to width ratio")

            # cell outlines
            ax.imshow(
                self._masked_cell_outlines(im_junction, segmentation_mask),
                alpha=self.params.alpha_cell_outline,
                cmap="gray_r",
            )

        # plot major and minor axis
        for _, row in collection.get_properties_by_img_name(img_name).iterrows():
            if inst_nuclei_mask is not None:
                # plot orientation degree
                Plotter._add_single_cell_length_to_width_ratio_axis(
                    ax[0],
                    row["cell_X"],
                    row["cell_Y"],
                    row["cell_shape_orientation_rad"],
                    row["cell_major_axis_length"],
                    row["cell_minor_axis_length"],
                    row["cell_length_to_width_ratio"],
                    self.params.fontsize_text_annotations,
                    self.params.font_color,
                    self.params.marker_size,
                    decimals=1,
                )

                # plot orientation degree nucleus
                Plotter._add_single_cell_length_to_width_ratio_axis(
                    ax[1],
                    row["nuc_X"],
                    row["nuc_Y"],
                    row["nuc_shape_orientation_rad"],
                    row["nuc_major_axis_length"],
                    row["nuc_minor_axis_length"],
                    row["cell_length_to_width_ratio"],
                    self.params.fontsize_text_annotations,
                    self.params.font_color,
                    self.params.marker_size,
                    decimals=1,
                )
            else:
                Plotter._add_single_cell_length_to_width_ratio_axis(
                    ax,
                    row["cell_X"],
                    row["cell_Y"],
                    row["cell_shape_orientation_rad"],
                    row["cell_major_axis_length"],
                    row["cell_minor_axis_length"],
                    row["cell_length_to_width_ratio"],
                    self.params.fontsize_text_annotations,
                    self.params.font_color,
                    self.params.marker_size,
                    decimals=1,
                )

        # set title and ax limits
        if inst_nuclei_mask is not None:
            add_title(
                ax[0],
                "cell elongation",
                im_junction.data,
                self.params.show_graphics_axis,
            )
            add_title(
                ax[1],
                "nuclei elongation",
                im_junction.data,
                self.params.show_graphics_axis,
            )
            axes = [ax[0], ax[1]]
        else:
            add_title(
                ax, "cell elongation", im_junction.data, self.params.show_graphics_axis
            )
            axes = [ax]

        self._finish_plot(
            fig,
            collection.get_out_path_by_name(img_name),
            img_name,
            "_elongation",
            axes,
            pixel_to_micron_ratio,
            close,
        )

        return fig, axes

    def plot_circularity(
        self, collection: PropertiesCollection, img_name: str, close: bool = False
    ):
        """Plot the circularity of cells (and nuclei) in a specific image in the collection.

        Args:
            collection:
                The collection containing the features
            img_name:
                The name of the image to plot
            close:
                whether to close the figure after saving

        """
        img = collection.get_image_by_img_name(img_name)
        assert img.segmentation is not None, "Segmentation is not available"
        assert img.junction is not None, "Junction channel not available"
        assert collection.dataset.empty is False, "No data available"
        assert (
            img_name in pandas.Series(list(collection.dataset["filename"])).unique()
        ), "There seems to be no data for the image you selected"
        # assert feature
        assert "cell_circularity" in list(collection.dataset.columns), (
            "Features %s not available." % "cell_circularity"
        )

        im_junction = img.junction
        segmentation_mask = img.segmentation.segmentation_mask_connected
        inst_nuclei_mask = None
        if img.has_nuclei():
            assert (
                img.segmentation.segmentation_mask_nuclei is not None
            ), "Nuclei segmentation not available"
            inst_nuclei_mask = img.segmentation.segmentation_mask_nuclei

        pixel_to_micron_ratio = img.img_params.pixel_to_micron_ratio

        get_logger().info("Plotting: circularity")

        # figure and axes
        number_sub_figs = 1
        if inst_nuclei_mask is not None:
            number_sub_figs = 2

        fig, ax = self._get_figure(number_sub_figs)

        # get cell_circularity
        cell_circularity_vec = collection.get_properties_by_img_name(img_name)[
            "cell_circularity"
        ].values
        cell_circularity = segmentation_mask.relabel(cell_circularity_vec)

        # add cell (and nuclei) circularity to the figure
        if inst_nuclei_mask is not None:
            Plotter._add_values_0_1(
                fig,
                ax[0],
                im_junction,
                cell_circularity,
                self.params.alpha,
                "circularity",
            )
            # get nuclei circularity
            nuclei_circularity_vec = collection.get_properties_by_img_name(img_name)[
                "nuc_circularity"
            ].values
            nuclei_circularity = inst_nuclei_mask.relabel(nuclei_circularity_vec)

            get_logger().info(
                "Maximal nuclei circularity: %s" % str(np.max(nuclei_circularity.data))
            )
            get_logger().info(
                "Minimal nuclei circularity: %s" % str(np.min(nuclei_circularity.data))
            )

            Plotter._add_values_0_1(
                fig,
                ax[1],
                im_junction,
                nuclei_circularity,
                self.params.alpha,
                "circularity",
            )
            # show cell outlines
            outline = self._masked_cell_outlines(im_junction, segmentation_mask)
            ax[0].imshow(outline, alpha=self.params.alpha_cell_outline, cmap="gray_r")
            ax[1].imshow(outline, alpha=self.params.alpha_cell_outline, cmap="gray_r")
        else:
            Plotter._add_values_0_1(
                fig, ax, im_junction, cell_circularity, self.params.alpha, "circularity"
            )

            # show cell outlines
            ax.imshow(
                self._masked_cell_outlines(im_junction, segmentation_mask),
                alpha=self.params.alpha_cell_outline,
                cmap="gray_r",
            )

        plot_title = "cell circularity"
        if self.params.show_statistics:
            cell_circularity_values = collection.get_properties_by_img_name(img_name)[
                "cell_circularity"
            ].values
            cell_circularity_values = cell_circularity_values[
                ~np.isnan(cell_circularity_values)
            ]
            plot_title += "\n N: " + str(len(cell_circularity_values)) + ", "
            plot_title += (
                "mean: " + str(np.round(np.mean(cell_circularity_values), 2)) + ", "
            )
            plot_title += "std: " + str(np.round(np.std(cell_circularity_values), 2))

        # set title and ax limits
        if inst_nuclei_mask is not None:

            add_title(
                ax[0],
                plot_title,
                im_junction.data,
                self.params.show_graphics_axis,
            )

            plot_title = "nuclei circularity"
            if self.params.show_statistics:
                cell_circularity_values = collection.get_properties_by_img_name(
                    img_name
                )["nuc_circularity"].values
                cell_circularity_values = cell_circularity_values[
                    ~np.isnan(cell_circularity_values)
                ]
                plot_title += "\n N: " + str(len(cell_circularity_values)) + ", "
                plot_title += (
                    "mean: " + str(np.round(np.mean(cell_circularity_values), 2)) + ", "
                )
                plot_title += "std: " + str(
                    np.round(np.std(cell_circularity_values), 2)
                )

            add_title(
                ax[1],
                plot_title,
                im_junction.data,
                self.params.show_graphics_axis,
            )
            axes = [ax[0], ax[1]]
        else:
            add_title(ax, plot_title, im_junction.data, self.params.show_graphics_axis)
            axes = [ax]

        self._finish_plot(
            fig,
            collection.get_out_path_by_name(img_name),
            img_name,
            "_circularity",
            axes,
            pixel_to_micron_ratio,
            close,
        )

        return fig, axes

    def plot_marker_cue_intensity_ratio(
        self, collection: PropertiesCollection, img_name: str, close: bool = False
    ):
        """Plot the marker cue intensity ratios of a specific image in the collection.

        Args:
            collection:
                The collection containing the features
            img_name:
                The name of the image to plot
            close:
                whether to close the figure after saving

        """
        get_logger().info("Plotting: marker cue intensity ratios")

        img = collection.get_image_by_img_name(img_name)
        assert img.segmentation is not None, "Segmentation is not available"
        assert img.junction is not None, "Junction channel not available"
        assert img.marker is not None, "Marker channel not available"
        assert collection.dataset.empty is False, "No data available"
        assert (
            img_name in pandas.Series(list(collection.dataset["filename"])).unique()
        ), "There seems to be no data for the image you selected"
        # assert feature
        assert "marker_cue_directional_intensity_ratio" in list(
            collection.dataset.columns
        ), ("Features %s not available." % "marker_cue_directional_intensity_ratio")

        params = collection.get_runtime_params_by_img_name(img_name)

        im_junction = img.junction
        cell_mask = img.segmentation.segmentation_mask_connected

        mcdir = collection.get_properties_by_img_name(img_name)[
            "marker_cue_directional_intensity_ratio"
        ].values
        mcdir_mask = cell_mask.relabel(mcdir)

        mcuir = collection.get_properties_by_img_name(img_name)[
            "marker_cue_axial_intensity_ratio"
        ].values
        mcuir_mask = cell_mask.relabel(mcuir)

        pixel_to_micron_ratio = img.img_params.pixel_to_micron_ratio

        ax, fig = self._plot_cue_intensity_ratio(
            cell_mask, collection, img.marker, img_name, mcdir_mask, mcuir_mask, params
        )

        add_title(
            ax[0],
            "marker cue directional intensity ratio",
            im_junction.data,
            self.params.show_graphics_axis,
        )
        add_title(
            ax[1],
            "marker cue axial intensity ratio",
            im_junction.data,
            self.params.show_graphics_axis,
        )

        self._finish_plot(
            fig,
            collection.get_out_path_by_name(img_name),
            img_name,
            "_marker_ratio_method",
            ax,
            pixel_to_micron_ratio,
            close,
        )

        return fig, ax

    def plot_junction_cue_intensity_ratio(
        self, collection: PropertiesCollection, img_name: str, close: bool = False
    ):
        """Plot the junction cue intensity ratios of a specific image in the collection.

        Args:
            collection:
                The collection containing the features
            img_name:
                The name of the image to plot
            close:
                whether to close the figure after saving

        """
        get_logger().info("Plotting: junction cue intensity ratios")

        img = collection.get_image_by_img_name(img_name)
        assert img.segmentation is not None, "Segmentation is not available"
        assert img.junction is not None, "Junction channel not available"
        assert collection.dataset.empty is False, "No data available"
        assert (
            img_name in pandas.Series(list(collection.dataset["filename"])).unique()
        ), "There seems to be no data for the image you selected"
        # assert feature
        assert "junction_cue_directional_intensity_ratio" in list(
            collection.dataset.columns
        ), ("Features %s not available." % "junction_cue_directional_intensity_ratio")

        params = collection.get_runtime_params_by_img_name(img_name)

        im_junction = img.junction
        cell_mask = img.segmentation.segmentation_mask_connected

        jcdir = collection.get_properties_by_img_name(img_name)[
            "junction_cue_directional_intensity_ratio"
        ].values
        jcdir_mask = cell_mask.relabel(jcdir)

        jcuir = collection.get_properties_by_img_name(img_name)[
            "junction_cue_axial_intensity_ratio"
        ].values
        jcuir_mask = cell_mask.relabel(jcuir)

        pixel_to_micron_ratio = img.img_params.pixel_to_micron_ratio

        ax, fig = self._plot_cue_intensity_ratio(
            cell_mask, collection, im_junction, img_name, jcdir_mask, jcuir_mask, params
        )

        add_title(
            ax[0],
            "junction cue directional intensity ratio",
            im_junction.data,
            self.params.show_graphics_axis,
        )
        add_title(
            ax[1],
            "junction cue axial intensity ratio",
            im_junction.data,
            self.params.show_graphics_axis,
        )

        self._finish_plot(
            fig,
            collection.get_out_path_by_name(img_name),
            img_name,
            "_junction_ratio_method",
            ax,
            pixel_to_micron_ratio,
            close,
        )

        return fig, ax

    def _plot_cue_intensity_ratio(
        self,
        cell_mask,
        collection,
        im_junction,
        img_name,
        directional_mask,
        axial_mask,
        params,
    ):
        # figure and axes
        fig, ax = self._get_figure(2)
        # show junction and cell mask overlay
        ax[0].imshow(im_junction.data, cmap=plt.cm.gray, alpha=1.0)
        cax1 = ax[0].imshow(
            directional_mask.mask_background().data,
            cmap=cm.cm.balance,
            alpha=self.params.alpha,
        )
        ax[1].imshow(im_junction.data, cmap=plt.cm.gray, alpha=1.0)
        cax2 = ax[1].imshow(
            axial_mask.mask_background().data,
            cmap=cm.cm.balance,
            alpha=self.params.alpha,
        )

        for _, row in collection.get_properties_by_img_name(img_name).iterrows():
            x0 = row["cell_X"]
            y0 = row["cell_Y"]

            # plot center of cell
            ax[0].plot(x0, y0, ".b", markersize=5)
            ax[1].plot(x0, y0, ".b", markersize=5)

            a = [x0, y0]
            b = [x0 + row["cell_major_axis_length"], y0]  # lies horizontally
            ground_line = LineString([a, b])

            # rotate ground line based on cue_direction
            ground_line = rotate(ground_line, params.cue_direction, origin=a)

            d_lines, _ = get_divisor_lines(a, ground_line, 4)
            for d_line in d_lines:
                x1, y1 = (i[0] for i in d_line.boundary.centroid.coords.xy)
                ax[1].plot((x0, x1), (y0, y1), "--r", linewidth=0.5)

            d_lines, _ = get_divisor_lines(a, ground_line, 2)
            for d_line in d_lines:
                x1, y1 = (i[0] for i in d_line.boundary.centroid.coords.xy)
                ax[0].plot((x0, x1), (y0, y1), "--r", linewidth=0.5)

        add_colorbar(fig, cax1, ax[0], [-1, 0, 1], "directed intensity ratio")
        add_colorbar(fig, cax2, ax[1], [0, 0.5, 1], "undirected intensity ratio")

        # show cell outlines
        ax[0].imshow(
            self._masked_cell_outlines(im_junction, cell_mask),
            alpha=self.params.alpha_cell_outline,
            cmap="gray_r",
        )
        ax[1].imshow(
            self._masked_cell_outlines(im_junction, cell_mask),
            alpha=self.params.alpha_cell_outline,
            cmap="gray_r",
        )

        return ax, fig

    def plot_foi(
        self, collection: PropertiesCollection, img_name: str, close: bool = False
    ):
        """Plot the field of interest of a specific image in the collection.

        Args:
            collection:
                The collection containing the features
            img_name:
                The name of the image to plot
            close:
                whether to close the figure after saving

        """
        get_logger().info("Plotting: figure of interest")

        img = collection.get_image_by_img_name(img_name)
        assert img.segmentation is not None, "Segmentation is not available"
        assert img.junction is not None, "Junction channel not available"
        assert collection.dataset.empty is False, "No data available"
        assert (
            img_name in pandas.Series(list(collection.dataset["filename"])).unique()
        ), "There seems to be no data for the image you selected"

        im_junction = img.junction
        mask = img.segmentation.segmentation_mask_connected

        pixel_to_micron_ratio = img.img_params.pixel_to_micron_ratio

        single_cell_dataset = collection.dataset.loc[
            collection.dataset["filename"] == img_name
        ]
        foi_name = collection.get_runtime_params_by_img_name(
            img_name
        ).feature_of_interest
        # assert feature
        assert foi_name in list(single_cell_dataset.columns), (
            "Features %s not available." % foi_name
        )

        foi = single_cell_dataset[foi_name]

        # normalize for length_unit (e.g. microns)
        if self.params.length_unit == "microns":
            foi = foi * pixel_to_micron_ratio

        # figure and axes
        fig, ax = self._get_figure(1)
        ax.imshow(im_junction.data, cmap=plt.cm.gray, alpha=1.0)

        # plot the figure of interest
        m = np.copy(mask.data)
        for _, row in single_cell_dataset.iterrows():
            foi_val = row[foi_name]
            label = row["label"]

            m = np.where(mask.data == label, foi_val, m)

            ax.text(
                row["cell_X"],
                row["cell_Y"],
                str(np.round(row[foi_name], 1)),
                color=self.params.font_color,
                fontsize=self.params.fontsize_text_annotations,
            )

        cax = ax.imshow(
            np.ma.masked_where(m == 0, m), cmap=plt.cm.bwr, alpha=self.params.alpha
        )

        nanmin = np.nanmin(foi)
        nanmax = np.nanmax(foi)
        yticks = [nanmin, np.round(nanmin + (nanmax - nanmin) / 2, 1), nanmax]
        add_colorbar(fig, cax, ax, yticks, foi_name)

        # show cell outlines
        ax.imshow(
            self._masked_cell_outlines(im_junction, mask),
            alpha=self.params.alpha_cell_outline,
            cmap="gray_r",
        )

        # set title and ax limits
        add_title(
            ax, "feature of interest", im_junction.data, self.params.show_graphics_axis
        )

        self._finish_plot(
            fig,
            collection.get_out_path_by_name(img_name),
            img_name,
            "_foi",
            [ax],
            pixel_to_micron_ratio,
            close,
        )

        return fig, ax

    def plot_symmetry(
        self, collection: PropertiesCollection, img_name: str, close: bool = False
    ):
        """Plot the asymmetry of a specific image in the collection.

        Args:
            collection:
                The collection containing the features
            img_name:
                The name of the image to plot
            close:
                whether to close the figure after saving

        """
        img = collection.get_image_by_img_name(img_name)
        assert img.segmentation is not None, "Segmentation is not available"
        assert img.junction is not None, "Junction channel not available"
        assert collection.dataset.empty is False, "No data available"
        assert (
            img_name in pandas.Series(list(collection.dataset["filename"])).unique()
        ), "There seems to be no data for the image you selected"

        im_junction = img.junction
        segmentation_mask = img.segmentation.segmentation_mask_connected
        pixel_to_micron_ratio = img.img_params.pixel_to_micron_ratio

        get_logger().info("Plotting: symmetry")

        # figure and axes
        number_sub_figs = 1

        fig, ax = self._get_figure(number_sub_figs)

        # get cell_cue_direction_asymmetry
        cell_cue_direction_asymmetry_vec = collection.get_properties_by_img_name(
            img_name
        )["cell_cue_direction_symmetry"].values
        cell_cue_direction_asymmetry = segmentation_mask.relabel(
            cell_cue_direction_asymmetry_vec
        )

        Plotter._add_values_0_1(
            fig,
            ax,
            im_junction,
            cell_cue_direction_asymmetry,
            self.params.alpha,
            "symmetry",
        )

        # show cell outlines
        ax.imshow(
            self._masked_cell_outlines(im_junction, segmentation_mask),
            alpha=self.params.alpha_cell_outline,
            cmap="gray_r",
        )

        # add cue_direction orientation
        params = collection.get_runtime_params_by_img_name(img_name)
        for _, row in collection.get_properties_by_img_name(img_name).iterrows():
            x0 = row["cell_X"]
            y0 = row["cell_Y"]

            # plot center of cell
            ax.plot(x0, y0, ".b", markersize=5)

            a = [x0, y0]
            b = [x0 + row["cell_major_axis_length"], y0]  # lies horizontally
            ground_line = LineString([a, b])

            # rotate ground line based on cue_direction
            ground_line = rotate(ground_line, params.cue_direction, origin=a)

            d_lines, _ = get_divisor_lines(a, ground_line, 2)
            for d_line in d_lines:
                x1, y1 = (i[0] for i in d_line.boundary.centroid.coords.xy)
                ax.plot((x0, x1), (y0, y1), "--r", linewidth=0.5)

        plot_title = "cell symmetry"
        if self.params.show_statistics:
            cell_cue_direction_asymmetry_val = cell_cue_direction_asymmetry_vec[
                ~np.isnan(cell_cue_direction_asymmetry_vec)
            ]
            plot_title += "\n N: " + str(len(cell_cue_direction_asymmetry_val)) + ", "
            plot_title += (
                "mean: "
                + str(np.round(np.mean(cell_cue_direction_asymmetry_val), 2))
                + ", "
            )
            plot_title += "std: " + str(
                np.round(np.std(cell_cue_direction_asymmetry_val), 2)
            )

        add_title(ax, plot_title, im_junction.data, self.params.show_graphics_axis)
        axes = [ax]

        self._finish_plot(
            fig,
            collection.get_out_path_by_name(img_name),
            img_name,
            "_symmetry",
            axes,
            pixel_to_micron_ratio,
            close,
        )

    def plot_shape_orientation(
        self, collection: PropertiesCollection, img_name: str, close: bool = False
    ):
        """Plot the orientation of a specific image in the collection.

        Args:
            collection:
                The collection containing the features
            img_name:
                The name of the image to plot
            close:
                whether to close the figure after saving

        """
        img = collection.get_image_by_img_name(img_name)
        assert img.segmentation is not None, "Segmentation is not available"
        assert img.junction is not None, "Junction channel not available"
        assert collection.dataset.empty is False, "No data available"
        assert (
            img_name in pandas.Series(list(collection.dataset["filename"])).unique()
        ), "There seems to be no data for the image you selected"

        im_junction = img.junction
        instance_segmentation_con = img.segmentation.segmentation_mask_connected

        inst_nuclei_mask = None
        if img.has_nuclei():
            assert (
                img.segmentation.segmentation_mask_nuclei is not None
            ), "Nuclei segmentation is not available"
            inst_nuclei_mask = img.segmentation.segmentation_mask_nuclei

        pixel_to_micron_ratio = img.img_params.pixel_to_micron_ratio
        r_params = collection.get_runtime_params_by_img_name(img_name)
        cue_direction = r_params.cue_direction

        get_logger().info("Plotting: orientation")

        # figure and axes
        number_sub_figs = 1
        if inst_nuclei_mask is not None:
            number_sub_figs = 2

        fig, ax = self._get_figure(number_sub_figs)

        # get cell_orientation
        cell_orientation_vec = collection.get_properties_by_img_name(img_name)[
            "cell_shape_orientation_deg"
        ].values
        cell_orientation = instance_segmentation_con.relabel(cell_orientation_vec)

        # add cell (and nuclei) orientation to the figure
        if inst_nuclei_mask is not None:
            Plotter._add_cell_orientation(
                fig, ax[0], im_junction, cell_orientation, self.params.alpha
            )

            # get nuclei orientation
            nuc_shape_orientation_rad_vector = collection.get_properties_by_img_name(
                img_name
            )["nuc_shape_orientation_rad"].values
            nuc_shape_orientation_deg_vector = (
                nuc_shape_orientation_rad_vector * 180.0 / np.pi
            )
            nuclei_orientation = inst_nuclei_mask.relabel(
                nuc_shape_orientation_deg_vector
            )

            get_logger().info(
                "Maximal nuclei orientation: %s" % str(np.max(nuclei_orientation.data))
            )
            get_logger().info(
                "Minimal nuclei orientation: %s" % str(np.min(nuclei_orientation.data))
            )

            Plotter._add_nuclei_orientation(
                fig, ax[1], im_junction, nuclei_orientation, self.params.alpha
            )

            # cell outlines
            ax[0].imshow(
                self._masked_cell_outlines(im_junction, instance_segmentation_con),
                alpha=self.params.alpha_cell_outline,
                cmap="gray_r",
            )

            # cell outlines
            ax[1].imshow(
                self._masked_cell_outlines(im_junction, instance_segmentation_con),
                alpha=self.params.alpha_cell_outline,
                cmap="gray_r",
            )
        else:
            Plotter._add_cell_orientation(
                fig, ax, im_junction, cell_orientation, self.params.alpha
            )
            # cell outlines
            ax.imshow(
                self._masked_cell_outlines(im_junction, instance_segmentation_con),
                alpha=self.params.alpha_cell_outline,
                cmap="gray_r",
            )

        # plot major and minor axis
        if self.params.show_polarity_angles:
            for _, row in collection.get_properties_by_img_name(img_name).iterrows():
                if inst_nuclei_mask is not None:
                    # plot orientation degree
                    Plotter._add_single_cell_orientation_degree_axis(
                        ax[0],
                        row["cell_X"],
                        row["cell_Y"],
                        row["cell_shape_orientation_rad"],
                        row["cell_major_axis_length"],
                        row["cell_minor_axis_length"],
                        self.params.fontsize_text_annotations,
                        self.params.font_color,
                        self.params.marker_size,
                    )

                    # plot orientation degree nucleus
                    Plotter._add_single_cell_orientation_degree_axis(
                        ax[1],
                        row["nuc_X"],
                        row["nuc_Y"],
                        row["nuc_shape_orientation_rad"],
                        row["nuc_major_axis_length"],
                        row["nuc_minor_axis_length"],
                        self.params.fontsize_text_annotations,
                        self.params.font_color,
                        self.params.marker_size,
                    )
                else:
                    # plot orientation degree
                    Plotter._add_single_cell_orientation_degree_axis(
                        ax,
                        row["cell_X"],
                        row["cell_Y"],
                        row["cell_shape_orientation_rad"],
                        row["cell_major_axis_length"],
                        row["cell_minor_axis_length"],
                        self.params.fontsize_text_annotations,
                        self.params.font_color,
                        self.params.marker_size,
                    )

        plot_title = "cell shape orientation"

        if self.params.show_statistics:
            angles = np.array(
                collection.get_properties_by_img_name(img_name)[
                    "cell_shape_orientation_rad"
                ]
            )
            cue_direction_rad = np.deg2rad(cue_direction)
            alpha_m, R, c = compute_polarity_index(
                angles, cue_direction=cue_direction_rad, stats_mode="axial"
            )
            plot_title += "\n N: " + str(len(angles)) + ", "
            plot_title += "mean angle: " + str(np.round(alpha_m, 2)) + "°, "
            plot_title += "PI: " + str(np.round(R, 2)) + ","
            plot_title += "\n polarity cue: " + str(np.round(cue_direction, 2)) + "°, "
            plot_title += "c: " + str(np.round(c, 2)) + ", "
            plot_title += "V: " + str(np.round(R * c, 2))

        # set title and ax limits
        if inst_nuclei_mask is not None:
            add_title(
                ax[0],
                plot_title,
                im_junction.data,
                self.params.show_graphics_axis,
            )
            plot_title_nuc = "nuclei shape orientation"

            if self.params.show_statistics:
                angles = np.array(
                    collection.get_properties_by_img_name(img_name)[
                        "nuc_shape_orientation_rad"
                    ]
                )
                cue_direction_rad = np.deg2rad(cue_direction)
                alpha_m, R, c = compute_polarity_index(
                    angles, cue_direction=cue_direction_rad, stats_mode="axial"
                )
                plot_title += "\n N: " + str(len(angles)) + ", "
                plot_title += "mean angle: " + str(np.round(alpha_m, 2)) + "°, "
                plot_title_nuc += "PI: " + str(np.round(R, 2)) + ","
                plot_title_nuc += "\n polarity cue: " + str(np.round(c, 2)) + "°, "
                plot_title_nuc += "c: " + str(np.round(c, 2)) + ", "
                plot_title_nuc += "V: " + str(np.round(R * c, 2))

            add_title(
                ax[1],
                plot_title_nuc,
                im_junction.data,
                self.params.show_graphics_axis,
            )
            axes = [ax[0], ax[1]]
        else:
            add_title(
                ax,
                plot_title,
                im_junction.data,
                self.params.show_graphics_axis,
            )
            axes = [ax]

        self._finish_plot(
            fig,
            collection.get_out_path_by_name(img_name),
            img_name,
            "_shape_orientation",
            axes,
            pixel_to_micron_ratio,
            close,
        )

        return fig, ax

    def plot_single_cell_centered_masks(
        self, collection: PropertiesCollection, img_name: str, close: bool = False
    ):
        """Plot single cell centered masks.

         This will produce for each single cell in an image a figure with 4 subplots:
            - centered_junction_mask
            - centered_nuc_mask
            - centered_organelle_mask
            - centered_cytosol_mask

        Args:
            collection:
                The collection containing the features
            img_name:
                The name of the image to plot
            close:
                whether to close the figure after saving

        """
        assert (
            collection.sc_img_dict is not None or collection.sc_img_dict != {}
        ), "Single cell images are not available!"

        get_logger().info("Plotting: single cell centered masks")

        try:
            single_cells_list = collection.sc_img_dict[img_name]
        except KeyError:
            get_logger().warning(
                "No single cell images for image %s available! "
                'Did you specify option "save_sc_image" in the Runtime Parameters?'
                % img_name
            )
            return

        plt_scalebar = self.params.show_scalebar
        self.params.show_scalebar = False

        # loop
        for idx, single_cell in enumerate(single_cells_list):
            centered_masks = {
                "centered_cell_mask": single_cell.center_mask(single_cell.cell_mask),
                "centered_membrane_mask": single_cell.center_mask(
                    single_cell.cell_membrane_mask
                ),
            }

            # junction mask
            if single_cell.has_junction():
                centered_masks["centered_junction_mask"] = single_cell.center_mask(
                    single_cell.junction_mask
                )

            # nuclei mask
            if single_cell.has_nuclei():
                centered_masks["centered_nuc_mask"] = single_cell.center_mask(
                    single_cell.nucleus_mask
                )

            # organelle mask
            if single_cell.has_organelle():
                centered_masks["centered_organelle_mask"] = single_cell.center_mask(
                    single_cell.organelle_mask
                )

            # marker mask
            if single_cell.has_marker():
                centered_masks["centered_cytosol_mask"] = single_cell.center_mask(
                    single_cell.cytosol_mask
                )

            if self.params.plot_sc_partitions:
                half_masks = single_cell.half_mask(
                    collection.get_runtime_params_by_img_name(img_name).cue_direction
                )
                quad_masks = single_cell.quarter_mask(
                    collection.get_runtime_params_by_img_name(img_name).cue_direction
                )
                centered_masks["centered_half_mask_r"] = single_cell.center_mask(
                    half_masks[0]
                )
                centered_masks["centered_half_mask_l"] = single_cell.center_mask(
                    half_masks[1]
                )
                centered_masks["quadrant_mask_r"] = single_cell.center_mask(
                    quad_masks[0]
                )
                centered_masks["quadrant_mask_t"] = single_cell.center_mask(
                    quad_masks[1]
                )
                centered_masks["quadrant_mask_l"] = single_cell.center_mask(
                    quad_masks[2]
                )
                centered_masks["quadrant_mask_b"] = single_cell.center_mask(
                    quad_masks[3]
                )

            # plot
            fig, ax = self._get_figure(len(centered_masks))

            for i, (mask_name, mask) in enumerate(centered_masks.items()):
                ax[i].imshow(mask.data, cmap="gray")
                add_title(ax[i], mask_name, mask.data, self.params.show_graphics_axis)

            self._finish_plot(
                fig,
                collection.get_out_path_by_name(img_name),
                img_name,
                "_single_cell_centered_masks_%s" % idx,
                ax,
                1,
                close,
            )

        self.params.show_scalebar = plt_scalebar

    def plot_threshold_segmentation_mask(
        self, collection: PropertiesCollection, img_name: str, close: bool = False
    ):
        """Show the processed segmentation mask.

        Args:
            collection:
                The collection containing the features
            img_name:
                The name of the image to plot
            close:
                whether to close the figure after saving

        """
        get_logger().info("Plotting: processed segmentation mask")

        img = collection.get_image_by_img_name(img_name)
        assert img.segmentation is not None, "Segmentation is not available"

        masks = []
        titles = []
        num_fig = 2
        if img.segmentation.segmentation_mask_nuclei is not None:
            masks.append(img.segmentation.segmentation_mask_nuclei)
            titles.append("threshold segmentation mask nuclei")
            num_fig += 1
        if img.segmentation.segmentation_mask_organelle is not None:
            masks.append(img.segmentation.segmentation_mask_organelle)
            titles.append("threshold segmentation mask organelle")
            num_fig += 1
        if img.segmentation.segmentation_mask_junction is not None:
            masks.append(img.segmentation.segmentation_mask_junction)
            titles.append("threshold segmentation mask junction")
            num_fig += 1

        fig, ax = self._get_figure(num_fig)

        # ignore background
        _mask = np.where(
            img.segmentation.segmentation_mask.data > 0,
            img.segmentation.segmentation_mask.data,
            np.nan,
        )
        _mask = self._rand_labels(_mask)

        # plot unconnected segmentation mask
        ax[0].imshow(_mask, cmap=plt.cm.gist_rainbow)
        add_title(
            ax[0],
            "threshold segmentation mask",
            img.segmentation.segmentation_mask.data,
            self.params.show_graphics_axis,
        )

        # ignore background
        _mask_con = np.where(
            img.segmentation.segmentation_mask_connected.data > 0,
            img.segmentation.segmentation_mask_connected.data,
            np.nan,
        )
        _mask_con = self._rand_labels(_mask_con)

        # plot connected segmentation mask
        ax[1].imshow(_mask_con, cmap=plt.cm.gist_rainbow)
        add_title(
            ax[1],
            "connected threshold segmentation mask",
            img.segmentation.segmentation_mask_connected.data,
            self.params.show_graphics_axis,
        )

        # plot optional masks
        for i, (mask, title) in enumerate(zip(masks, titles)):
            # ignore background
            mask = np.where(mask.data > 0, mask.data, np.nan)
            mask_ = self._rand_labels(mask)

            ax[i + 2].imshow(mask_, cmap=plt.cm.gist_rainbow)
            add_title(ax[i + 2], title, mask_, self.params.show_graphics_axis)

        self._finish_plot(
            fig,
            collection.get_out_path_by_name(img_name),
            img_name,
            "_threshold_segmentation_mask",
            ax,
            img.img_params.pixel_to_micron_ratio,
            close,
        )

        return fig, ax

    def _finish_plot(
        self,
        fig,
        output_path,
        img_name,
        output_suffix,
        axes,
        pixel_to_micron_ratio,
        close=False,
        image=None,
    ):
        # plot scale bar for this figure
        if self.params.show_scalebar:
            for ax in axes:
                add_scalebar(
                    ax,
                    self.params.length_scalebar_microns,
                    pixel_to_micron_ratio,
                    int(self.params.length_scalebar_microns / 2),
                    self.params.font_color,
                )

        # save output & close
        save_current_fig(
            self.params.graphics_output_format,
            output_path,
            img_name,
            output_suffix,
            image=image,
        )

        # close figure
        if close:
            plt.close(fig)

    def plot_collection(self, collection: PropertiesCollection, close: bool = False):
        """Plot all features of all images in the collection.

        Args:
            collection:
                the collection to plot
            close:
                whether to close the figure after saving

        """
        for key in collection.img_dict.keys():
            img = collection.get_image_by_img_name(key)
            r_params = collection.get_runtime_params_by_img_name(key)

            if self.params.plot_threshold_masks:
                self.plot_threshold_segmentation_mask(collection, key, close)

            if self.params.plot_polarity and img.has_nuclei() and img.has_organelle():
                if r_params.extract_polarity_features:
                    self.plot_organelle_polarity(collection, key, close)
                    if img.has_nuclei():

                        self.plot_nuc_displacement_orientation(collection, key, close)

            if self.params.plot_marker and img.has_marker():
                if r_params.extract_intensity_features:
                    self.plot_marker_expression(collection, key, close)

                if r_params.extract_polarity_features:
                    self.plot_marker_polarity(collection, key, close)

                if img.has_nuclei() and r_params.extract_polarity_features:
                    self.plot_marker_nucleus_orientation(collection, key, close)

                if self.params.plot_ratio_method and r_params.extract_polarity_features:
                    if r_params.extract_morphology_features:
                        self.plot_marker_cue_intensity_ratio(collection, key, close)
                    else:
                        get_logger().warning(
                            "Cannot cue intensity ratio without morphology features."
                        )

            if self.params.plot_junctions and img.has_junction():
                if r_params.extract_morphology_features:
                    self.plot_junction_features(collection, key, close)
                    self.plot_corners(collection, key, close)

                if r_params.extract_polarity_features:
                    self.plot_junction_polarity(collection, key, close)

            if self.params.plot_elongation:
                if r_params.extract_morphology_features:
                    if r_params.extract_polarity_features:
                        self.plot_length_to_width_ratio(collection, key, close)
                    else:
                        get_logger().warning(
                            "Cannot plot elongation without polarity features."
                        )
                # self.plot_eccentricity(collection, key, close)

            if self.params.plot_symmetry and r_params.extract_polarity_features:
                if r_params.extract_morphology_features:
                    self.plot_symmetry(collection, key, close)
                else:
                    get_logger().warning(
                        "Cannot plot elongation without morphology features."
                    )

            if self.params.plot_circularity and r_params.extract_morphology_features:
                self.plot_circularity(collection, key, close)

            if self.params.plot_ratio_method and r_params.extract_polarity_features:
                if r_params.extract_morphology_features:
                    self.plot_junction_cue_intensity_ratio(collection, key, close)
                else:
                    get_logger().warning(
                        "Cannot plot cue intensity ratio without morphology features."
                    )

            if (
                self.params.plot_shape_orientation
                and r_params.extract_polarity_features
            ):
                if r_params.extract_morphology_features:
                    self.plot_shape_orientation(collection, key, close)
                else:
                    get_logger().warning(
                        "Cannot plot shape orientation without morphology features."
                    )

            if self.params.plot_foi:
                if r_params.extract_group_features:
                    # check if feature of interest is available
                    if r_params.feature_of_interest in collection.dataset.columns:
                        self.plot_foi(collection, key, close)
                    else:
                        get_logger().warning(
                            'Cannot plot feature of interest "%s". '
                            "Did you enable the respective feature class for extraction?"
                            % r_params.feature_of_interest
                        )

            if self.params.plot_sc_image:
                self.plot_single_cell_centered_masks(collection, key, close)

    @staticmethod
    def _add_single_cell_orientation_degree_axis(
        ax,
        y0,
        x0,
        orientation,
        major_axis_length,
        minor_axis_length,
        fontsize=3,
        font_color="w",
        markersize=2,
    ):
        (
            x1_ma,
            x1_mi,
            x2_ma,
            x2_mi,
            y1_ma,
            y1_mi,
            y2_ma,
            y2_mi,
        ) = Plotter._calc_single_cell_axis_orientation_vector(
            x0, y0, orientation, major_axis_length, minor_axis_length
        )
        orientation_degree = 180.0 * orientation / np.pi

        ax.plot((y1_ma, y2_ma), (x1_ma, x2_ma), "--w", linewidth=0.5)
        ax.plot((y1_mi, y2_mi), (x1_mi, x2_mi), "--w", linewidth=0.5)
        ax.plot(y0, x0, ".b", markersize=markersize)
        ax.text(
            y0,
            x0,
            str(int(np.round(orientation_degree, 0))),
            color=font_color,
            fontsize=fontsize,
        )

    @staticmethod
    def _add_nuclei_orientation(
        fig,
        ax,
        im_junction: BioMedicalChannel,
        nuclei_orientation: BioMedicalInstanceSegmentationMask,
        alpha,
    ):
        v_min = 0.0
        v_max = 180.0
        yticks = [0.0, 45.0, 90.0, 135.0, 180.0]

        ax.imshow(im_junction.data, cmap=plt.cm.gray, alpha=1.0)

        # show nuclei orientation everywhere but background label

        cax_1 = ax.imshow(
            nuclei_orientation.mask_background().data,
            cmap=cm.cm.phase,
            vmin=v_min,
            vmax=v_max,
            alpha=alpha,
        )

        # colorbar
        add_colorbar(fig, cax_1, ax, yticks, "shape orientation (degree)")

    @staticmethod
    def _rand_labels(mask):
        """Randomly assign labels to the mask."""
        # color each cell randomly
        cell_idx = np.unique(mask)
        cell_idx = np.delete(cell_idx, 0)
        mask_ = np.copy(mask)
        new_col = np.copy(cell_idx)
        np.random.seed(42)  # set seed for reproducibility
        np.random.shuffle(new_col)
        for i in range(len(cell_idx)):
            mask_[mask == cell_idx[i]] = new_col[i]
        return mask_

    @staticmethod
    def _add_cell_orientation(
        fig,
        ax,
        im_junction: BioMedicalChannel,
        cell_orientation: BioMedicalInstanceSegmentationMask,
        alpha,
    ):
        v_min = 0.0
        v_max = 180.0
        yticks = [0.0, 45.0, 90.0, 135.0, 180.0]

        ax.imshow(im_junction.data, cmap=plt.cm.gray, alpha=1.0)

        # show cell_orientation everywhere but background label
        cax = ax.imshow(
            cell_orientation.mask_background().data,
            cmap=cm.cm.phase,
            vmin=v_min,
            vmax=v_max,
            alpha=alpha,
        )

        # colorbar
        add_colorbar(fig, cax, ax, yticks, "shape orientation (degree)")

    @staticmethod
    def _calc_single_cell_axis_orientation_vector(
        x, y, orientation, major_axis_length, minor_axis_length
    ):
        x1_major = x + math.sin(orientation) * 0.5 * major_axis_length
        y1_major = y - math.cos(orientation) * 0.5 * major_axis_length
        x2_major = x - math.sin(orientation) * 0.5 * major_axis_length
        y2_major = y + math.cos(orientation) * 0.5 * major_axis_length

        x1_minor = x - math.cos(orientation) * 0.5 * minor_axis_length
        y1_minor = y - math.sin(orientation) * 0.5 * minor_axis_length
        x2_minor = x + math.cos(orientation) * 0.5 * minor_axis_length
        y2_minor = y + math.sin(orientation) * 0.5 * minor_axis_length

        return [
            x1_major,
            x1_minor,
            x2_major,
            x2_minor,
            y1_major,
            y1_minor,
            y2_major,
            y2_minor,
        ]

    @staticmethod
    def _add_values_0_1(
        fig,
        ax,
        im_junction: BioMedicalChannel,
        cell_circularity: BioMedicalInstanceSegmentationMask,
        alpha,
        label,
    ):
        v_min = 0.0
        v_max = 1.0
        yticks = [0.0, 0.5, 1.0]

        ax.imshow(im_junction.data, cmap=plt.cm.gray, alpha=1.0)

        # show cell_circularity everywhere but background label
        cax_0 = ax.imshow(
            cell_circularity.mask_background().data,
            cmap=plt.cm.bwr,
            vmin=v_min,
            vmax=v_max,
            alpha=alpha,
        )

        # colorbar
        add_colorbar(fig, cax_0, ax, yticks, label)

    @staticmethod
    def _add_single_cell_length_to_width_ratio_axis(
        ax,
        y0,
        x0,
        orientation,
        major_axis_length,
        minor_axis_length,
        feature_value,
        fontsize=3,
        font_color="w",
        markersize=2,
        decimals=2,
    ):
        (
            x1_ma,
            x1_mi,
            x2_ma,
            x2_mi,
            y1_ma,
            y1_mi,
            y2_ma,
            y2_mi,
        ) = Plotter._calc_single_cell_axis_orientation_vector(
            x0, y0, orientation, major_axis_length, minor_axis_length
        )

        ax.plot((y1_ma, y2_ma), (x1_ma, x2_ma), "--w", linewidth=0.5)
        ax.plot((y1_mi, y2_mi), (x1_mi, x2_mi), "--w", linewidth=0.5)
        ax.plot(y0, x0, ".b", markersize=markersize)
        ax.text(
            y0,
            x0,
            str(np.round(feature_value, decimals)),
            color=font_color,
            fontsize=fontsize,
        )
