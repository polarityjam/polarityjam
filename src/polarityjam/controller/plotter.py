import json
import math
import os
from pathlib import Path
from typing import List, Union

import cmocean as cm
import matplotlib
import numpy as np
import pandas
from matplotlib import pyplot as plt
from shapely.geometry import LineString

from polarityjam.compute.shape import get_divisor_lines
from polarityjam.model.collection import PropertiesCollection
from polarityjam.model.image import BioMedicalChannel
from polarityjam.model.masks import (BioMedicalInstanceSegmentationMask,
                                     BioMedicalMask)
from polarityjam.model.parameter import ImageParameter, PlotParameter
from polarityjam.polarityjam_logging import get_logger
from polarityjam.vizualization.plot import (add_colorbar, add_scalebar,
                                            add_title, add_vector,
                                            save_current_fig)


class Plotter:
    """Plotter class"""

    def __init__(self, params: PlotParameter):
        self.params = params
        self.set_figure_dpi()

    def set_figure_dpi(self):
        """Set figure dpi"""
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

            single_cell_mask = cell_mask.get_single_instance_maks(cell_label)
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
                self.params.membrane_thickness
            )
            inlines_mem = np.where(
                np.logical_and(outline_mem.data, inlines_mem.data < intensity_mem),
                intensity_mem,
                inlines_mem.data,
            )

            # nuclei marker intensity
            if nuclei_mask is not None:
                single_nucleus_mask = nuclei_mask.get_single_instance_maks(cell_label)
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

    def _masked_cell_outlines(
            self,
            channel: BioMedicalChannel,
            instance_seg_mask: BioMedicalInstanceSegmentationMask,
    ) -> np.ndarray:
        # cell outlines
        outlines_cells = BioMedicalMask.empty(channel.data.shape)
        for cell_label in instance_seg_mask.get_labels():
            single_cell_mask = instance_seg_mask.get_single_instance_maks(cell_label)
            outline_cell = single_cell_mask.get_outline_from_mask(
                self.params.outline_width
            )
            outlines_cells = outlines_cells.operation(outline_cell, np.logical_or)

        # convert cell outlines to image
        outlines_cells_rgba = outlines_cells.to_instance_mask().mask_background()
        outlines_cells_rgba = np.dstack([outlines_cells_rgba.data] * 3)

        return outlines_cells_rgba

    def plot_channels(
            self,
            seg_img: np.ndarray,
            seg_img_params: ImageParameter,
            output_path: Union[str, Path],
            filename: Union[str, Path],
            close=False,
    ):
        """Plots the separate channels from the input image given, based on its parameters.

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

        """
        get_logger().info("Plotting: input channels")

        filename, _ = os.path.splitext(os.path.basename(filename))

        if (
                seg_img_params.channel_junction is not None
                and seg_img_params.channel_nucleus is not None
        ):
            fig, ax = self._get_figure(2)

            # junction channel
            c_junction = seg_img_params.channel_junction
            ax[0].imshow(seg_img[c_junction, :, :])
            add_title(
                ax[0],
                "junction channel",
                seg_img[c_junction, :, :],
                self.params.show_graphics_axis,
            )

            # nucleus channel
            c_nucleus = seg_img_params.channel_nucleus
            ax[1].imshow(seg_img[c_nucleus, :, :])
            add_title(
                ax[1],
                "nuclei channel",
                seg_img[c_nucleus, :, :],
                self.params.show_graphics_axis,
            )

            axes = [ax[0], ax[1]]
        else:
            fig, ax = self._get_figure(1)

            # first channel
            ax.imshow(seg_img[:, :])
            add_title(
                ax, "first channel", seg_img[:, :], self.params.show_graphics_axis
            )
            axes = [ax]

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

    def plot_mask(
            self,
            mask: np.ndarray,
            seg_img: np.ndarray,
            seg_img_params: ImageParameter,
            output_path: Union[str, Path],
            filename: Union[str, Path],
            close: bool = False,
    ):
        """Plots the segmentation mask, together with the separate channels from the input image.


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
            close:
                whether to close the figure after saving

        """
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

        if (
                seg_img_params.channel_junction is not None
                and seg_img_params.channel_nucleus is not None
        ):
            fig, ax = self._get_figure(3)

            ax[0].imshow(seg_img[0, :, :])
            add_title(
                ax[0],
                "junction channel",
                seg_img[0, :, :],
                self.params.show_graphics_axis,
            )

            ax[1].imshow(seg_img[1, :, :])
            add_title(
                ax[1],
                "nuclei channel",
                seg_img[1, :, :],
                self.params.show_graphics_axis,
            )

            ax[2].imshow(seg_img[0, :, :])
            ax[2].imshow(mask_, cmap=plt.cm.gist_rainbow, alpha=0.5)
            add_title(
                ax[2], "segmentation", seg_img[0, :, :], self.params.show_graphics_axis
            )

            axes = [ax[0], ax[1], ax[2]]
        else:
            fig, ax = self._get_figure(2)

            s_img = seg_img[:, :]

            # ax 1
            ax[0].imshow(s_img)
            add_title(ax[0], "junction channel", s_img, self.params.show_graphics_axis)

            # ax 2
            ax[1].imshow(s_img)
            ax[1].imshow(mask_, cmap=plt.cm.gist_rainbow, alpha=0.5)
            add_title(ax[1], "segmentation", s_img, self.params.show_graphics_axis)

            axes = [ax[0], ax[1]]

        self._finish_plot(
            fig,
            output_path,
            filename,
            "_segmentation",
            axes,
            seg_img_params.pixel_to_micron_ratio,
            close,
        )

        return fig, axes

    def plot_organelle_polarity(
            self, collection: PropertiesCollection, img_name: str, close: bool = False
    ):
        """Plots the organelle polarity of a specific image in the collection

        Args:
            collection:
                The collection containing the features
            img_name:
                The name of the image to plot
            close:
                whether to close the figure after saving

        """
        img = collection.get_image_by_img_name(img_name)

        im_junction = img.junction.data
        con_inst_seg_mask = img.segmentation.segmentation_mask_connected
        inst_nuclei_mask = img.nucleus.get_mask_by_name("nuclei_mask_seg")
        inst_organelle_mask = img.organelle.get_mask_by_name("organelle_mask_seg")

        pixel_to_micron_ratio = img.img_params.pixel_to_micron_ratio

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
            alpha=0.5,
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

        # set title and ax limits
        add_title(
            ax,
            "organelle orientation",
            im_junction.data,
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
        """Plots the nucleus displacement orientation of a specific image in the collection

        Args:
            collection:
                The collection containing the features
            img_name:
                The name of the image to plot
            close:
                whether to close the figure after saving

        """
        img = collection.get_image_by_img_name(img_name)

        pixel_to_micron_ratio = img.img_params.pixel_to_micron_ratio

        get_logger().info("Plotting: marker nucleus polarity")

        # figure and axes
        fig, ax = self._get_figure(1)

        # resources image
        ax.imshow(img.junction.data, cmap=plt.cm.gray, alpha=1.0)

        # determine nucleus polarity_angle
        nuc_polarity_angle_vec = collection.get_properties_by_img_name(img_name)[
            "nuc_displacement_orientation_deg"
        ].values
        nuc_polarity_angle = img.segmentation.segmentation_mask.relabel(
            nuc_polarity_angle_vec
        )

        # plot polarity angle
        cax = ax.imshow(
            nuc_polarity_angle.mask_background().data,
            cmap=cm.cm.phase,
            vmin=0,
            vmax=360,
            alpha=0.5,
        )
        color_bar = fig.colorbar(cax, ax=ax, shrink=0.3)  # , extend='both')
        color_bar.set_label("polarity angle")
        color_bar.ax.set_yticks([0, 90, 180, 270, 360])

        nuclei_mask = img.nucleus.get_mask_by_name("nuclei_mask_seg").to_semantic_mask()

        # plot nuclei (blue)
        zero = np.zeros((img.junction.data.shape[0], img.junction.data.shape[1]))
        rgba_nuclei = np.dstack((zero, zero, nuclei_mask.data, nuclei_mask.data))
        ax.imshow(rgba_nuclei)

        # plot polarity vector
        for index, row in collection.get_properties_by_img_name(img_name).iterrows():
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

        # set title and ax limits
        add_title(
            ax,
            "nucleus displacement orientation",
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
        """Plots the marker expression of a specific image in the collection

        Args:
            collection:
                The collection containing the features
            img_name:
                The name of the image to plot
            close:
                whether to close the figure after saving

        """
        img = collection.get_image_by_img_name(img_name)

        im_marker = img.marker.data
        cell_mask = img.segmentation.segmentation_mask_connected
        single_cell_dataset = collection.dataset.loc[
            collection.dataset["filename"] == img_name
            ]

        nuclei_mask = None
        if img.img_params.channel_nucleus >= 0:
            nuclei_mask = img.nucleus.get_mask_by_name("nuclei_mask_seg")

        pixel_to_micron_ratio = img.img_params.pixel_to_micron_ratio

        get_logger().info("Plotting: marker expression")
        # figure and axes
        number_sub_figs = 2  # mean intensity cell, mean intensity membrane
        if nuclei_mask is not None:
            number_sub_figs = 3  # (optional) mean intensity nucleus

        fig, ax = self._get_figure(number_sub_figs)

        # plot marker intensity for all subplots
        for i in range(number_sub_figs):
            ax[i].imshow(im_marker, cmap=plt.cm.gray, alpha=1.0)

        outlines_cell, outlines_mem, outlines_nuc = self._get_inlines(
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
            cax_3 = ax[2].imshow(
                outlines_nuc_, plt.cm.bwr, alpha=0.75
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
        for index, row in single_cell_dataset.iterrows():
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
            ax[0], "mean intensity cell", im_marker.data, self.params.show_graphics_axis
        )
        add_title(
            ax[1],
            "mean intensity membrane",
            im_marker.data,
            self.params.show_graphics_axis,
        )
        if nuclei_mask is not None:
            add_title(
                ax[2],
                "mean intensity nucleus",
                im_marker.data,
                self.params.show_graphics_axis,
            )
            axes = [ax[0], ax[1], ax[2]]

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
        """Plots the marker polarity of a specific image in the collection

        Args:
            collection:
                The collection containing the features
            img_name:
                The name of the image to plot
            close:
                whether to close the figure after saving

        """
        img = collection.get_image_by_img_name(img_name)

        im_marker = img.marker
        cell_mask = img.segmentation.segmentation_mask_connected

        pixel_to_micron_ratio = img.img_params.pixel_to_micron_ratio

        get_logger().info("Plotting: marker polarity")

        # figure and axes
        fig, ax = self._get_figure(1)

        # plot marker intensity
        ax.imshow(im_marker.data, cmap=plt.cm.gray, alpha=1.0)

        # show cell outlines
        ax.imshow(self._masked_cell_outlines(im_marker, cell_mask), alpha=0.5)

        # add all polarity vectors
        for index, row in collection.get_properties_by_img_name(img_name).iterrows():
            add_vector(
                ax,
                row["cell_X"],
                row["cell_Y"],
                row["marker_centroid_X"],
                row["marker_centroid_Y"],
                self.params.marker_size,
                self.params.font_color,
            )

        add_title(ax, "marker polarity", im_marker.data, self.params.show_graphics_axis)

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
        """Plots the marker polarity of a specific image in the collection

        Args:
            collection:
                The collection containing the features
            img_name:
                The name of the image to plot
            close:
                whether to close the figure after saving

        """
        img = collection.get_image_by_img_name(img_name)

        im_junction = img.junction.data
        segmentation_mask = img.segmentation.segmentation_mask_connected
        inst_nuclei_mask = img.nucleus.get_mask_by_name("nuclei_mask_seg")

        pixel_to_micron_ratio = img.img_params.pixel_to_micron_ratio

        get_logger().info("Plotting: marker nucleus polarity")

        # figure and axes
        fig, ax = self._get_figure(1)

        # resources image
        ax.imshow(im_junction, cmap=plt.cm.gray, alpha=1.0)

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
            alpha=0.5,
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
        for index, row in collection.get_properties_by_img_name(img_name).iterrows():
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

        # set title and ax limits
        add_title(
            ax,
            "marker nucleus orientation",
            im_junction.data,
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

    def plot_junction_polarity(
            self, collection: PropertiesCollection, img_name: str, close: bool = False
    ):
        """Plots the junction polarity of a specific image in the collection

        Args:
            collection:
                The collection containing the features
            img_name:
                The name of the image to plot
            close:
                whether to close the figure after saving

        """
        img = collection.get_image_by_img_name(img_name)

        im_junction = img.junction
        cell_mask = img.segmentation.segmentation_mask_connected

        pixel_to_micron_ratio = img.img_params.pixel_to_micron_ratio

        get_logger().info("Plotting: junction polarity")

        # figure and axes
        fig, ax = self._get_figure(1)

        # plot marker intensity
        ax.imshow(im_junction.data, cmap=plt.cm.gray, alpha=1.0)

        # show cell outlines
        ax.imshow(self._masked_cell_outlines(im_junction, cell_mask), alpha=0.5)

        # add all polarity vectors
        for index, row in collection.get_properties_by_img_name(img_name).iterrows():
            add_vector(
                ax,
                row["cell_X"],
                row["cell_Y"],
                row["junction_centroid_X"],
                row["junction_centroid_Y"],
                self.params.marker_size,
                self.params.font_color,
            )

        add_title(
            ax, "junction polarity", im_junction.data, self.params.show_graphics_axis
        )

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
        """Plots the corners of a specific image in the collection

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
        pixel_to_micron_ratio = img.img_params.pixel_to_micron_ratio

        # plot marker intensity
        im_junction = img.junction
        cell_mask = img.segmentation.segmentation_mask_connected

        ax.imshow(im_junction.data, cmap=plt.cm.gray, alpha=1.0)

        for index, row in collection.dataset.loc[
            collection.dataset["filename"] == img_name
        ].iterrows():
            plt.scatter(
                np.array(json.loads(row["cell_corner_points"]))[:, 0],
                np.array(json.loads(row["cell_corner_points"]))[:, 1],
                [4] * len(np.array(json.loads(row["cell_corner_points"]))[:, 1]),
            )

        ax.imshow(self._masked_cell_outlines(im_junction, cell_mask), alpha=0.5)

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
        """Plots the eccentricity of a specific image in the collection

        Args:
            collection:
                The collection containing the features
            img_name:
                The name of the image to plot
            close:
                whether to close the figure after saving

        """
        img = collection.get_image_by_img_name(img_name)

        im_junction = img.junction
        segmentation_mask = img.segmentation.segmentation_mask_connected
        inst_nuclei_mask = None
        if img.img_params.channel_nucleus >= 0:
            inst_nuclei_mask = img.nucleus.get_mask_by_name("nuclei_mask_seg")

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
            Plotter._add_cell_eccentricity(fig, ax[0], im_junction, cell_eccentricity)
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

            Plotter._add_nuclei_eccentricity(
                fig, ax[1], im_junction, nuclei_eccentricity
            )
        else:
            Plotter._add_cell_eccentricity(fig, ax, im_junction, cell_eccentricity)

        # plot major and minor axis
        for index, row in collection.get_properties_by_img_name(img_name).iterrows():
            if inst_nuclei_mask is not None:
                # plot orientation degree
                Plotter._add_single_cell_eccentricity_axis(
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
                Plotter._add_single_cell_eccentricity_axis(
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
                Plotter._add_single_cell_eccentricity_axis(
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
            "_eccentricity",
            axes,
            pixel_to_micron_ratio,
            close,
        )

        return fig, axes

    def plot_marker_cue_intensity_ratio(
            self, collection: PropertiesCollection, img_name: str, close: bool = False
    ):
        """Plots the marker cue intensity ratios of a specific image in the collection

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
        params = collection.get_runtime_params_by_img_name(img_name)

        im_junction = img.junction
        cell_mask = img.segmentation.segmentation_mask_connected

        mcdir = collection.get_properties_by_img_name(img_name)[
            "marker_cue_directional_intensity_ratio"
        ].values
        mcdir_mask = cell_mask.relabel(mcdir)

        mcuir = collection.get_properties_by_img_name(img_name)[
            "marker_cue_undirectional_intensity_ratio"
        ].values
        mcuir_mask = cell_mask.relabel(mcuir)

        pixel_to_micron_ratio = img.img_params.pixel_to_micron_ratio

        ax, fig = self._plot_cue_intensity_ratio(
            cell_mask, collection, im_junction, img_name, mcdir_mask, mcuir_mask, params
        )

        add_title(
            ax[0],
            "marker cue directional intensity ratio",
            im_junction.data,
            self.params.show_graphics_axis,
        )
        add_title(
            ax[1],
            "marker cue undirectional intensity ratio",
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

    def plot_junction_cue_intensity_ratio(
            self, collection: PropertiesCollection, img_name: str, close: bool = False
    ):
        """Plots the junction cue intensity ratios of a specific image in the collection

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
        params = collection.get_runtime_params_by_img_name(img_name)

        im_junction = img.junction
        cell_mask = img.segmentation.segmentation_mask_connected

        jcdir = collection.get_properties_by_img_name(img_name)[
            "junction_cue_directional_intensity_ratio"
        ].values
        jcdir_mask = cell_mask.relabel(jcdir)

        jcuir = collection.get_properties_by_img_name(img_name)[
            "junction_cue_undirectional_intensity_ratio"
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
            "junction cue undirectional intensity ratio",
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
            undirectional_mask,
            params,
    ):
        # figure and axes
        fig, ax = self._get_figure(2)
        # show junction and cell mask overlay
        ax[0].imshow(im_junction.data, cmap=plt.cm.gray, alpha=1.0)
        cax1 = ax[0].imshow(directional_mask.mask_background().data, cmap=cm.cm.balance, alpha=0.5)
        ax[1].imshow(im_junction.data, cmap=plt.cm.gray, alpha=1.0)
        cax2 = ax[1].imshow(undirectional_mask.mask_background().data, cmap=cm.cm.balance, alpha=0.5)
        # show cell outlines
        ax[0].imshow(self._masked_cell_outlines(im_junction, cell_mask), alpha=0.5)
        ax[1].imshow(self._masked_cell_outlines(im_junction, cell_mask), alpha=0.5)
        for index, row in collection.get_properties_by_img_name(img_name).iterrows():
            x0 = row["cell_X"]
            y0 = row["cell_Y"]

            # plot center of cell
            ax[0].plot(x0, y0, ".b", markersize=5)
            ax[1].plot(x0, y0, ".b", markersize=5)

            a = [x0, y0]
            b = [x0 + row["cell_major_axis_length"], y0]  # lies horizontally
            ground_line = LineString([a, b])

            d_lines, _ = get_divisor_lines(a, params.cue_direction, ground_line, 4)
            for d_line in d_lines:
                x1, y1 = (i[0] for i in d_line.boundary.centroid.coords.xy)
                ax[1].plot((x0, x1), (y0, y1), "--r", linewidth=0.5)

            d_lines, _ = get_divisor_lines(a, params.cue_direction, ground_line, 2)
            for d_line in d_lines:
                x1, y1 = (i[0] for i in d_line.boundary.centroid.coords.xy)
                ax[0].plot((x0, x1), (y0, y1), "--r", linewidth=0.5)

        add_colorbar(fig, cax1, ax[0], [-1, 0, 1], "directed intensity ratio")
        add_colorbar(fig, cax2, ax[1], [0, 0.5, 1], "undirected intensity ratio")

        return ax, fig

    def plot_foi(
            self, collection: PropertiesCollection, img_name: str, close: bool = False
    ):
        """Plots the field of interest of a specific image in the collection

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

        im_junction = img.junction
        mask = img.segmentation.segmentation_mask_connected

        pixel_to_micron_ratio = img.img_params.pixel_to_micron_ratio

        single_cell_dataset = collection.dataset.loc[
            collection.dataset["filename"] == img_name
            ]
        foi_name = collection.get_runtime_params_by_img_name(
            img_name
        ).feature_of_interest
        foi = single_cell_dataset[foi_name]
        # figure and axes
        fig, ax = self._get_figure(1)
        ax.imshow(im_junction.data, cmap=plt.cm.gray, alpha=1)

        # plot the figure of interest
        m = np.copy(mask.data)
        for index, row in single_cell_dataset.iterrows():
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

        cax = ax.imshow(np.ma.masked_where(m == 0, m), cmap=plt.cm.bwr, alpha=0.8)

        min = np.nanmin(foi)
        max = np.nanmax(foi)
        yticks = [min, np.round(min + (max - min) / 2, 1), max]
        add_colorbar(fig, cax, ax, yticks, foi_name)

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

    def plot_orientation(
            self, collection: PropertiesCollection, img_name: str, close: bool = False
    ):
        """Plots the orientation of a specific image in the collection

        Args:
            collection:
                The collection containing the features
            img_name:
                The name of the image to plot
            close:
                whether to close the figure after saving

        """
        img = collection.get_image_by_img_name(img_name)
        im_junction = img.junction
        instance_segmentation_con = img.segmentation.segmentation_mask_connected

        inst_nuclei_mask = None
        if img.has_nuclei():
            inst_nuclei_mask = img.nucleus.get_mask_by_name("nuclei_mask_seg")

        pixel_to_micron_ratio = img.img_params.pixel_to_micron_ratio

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
            Plotter._add_cell_orientation(fig, ax[0], im_junction, cell_orientation)

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

            Plotter._add_nuclei_orientation(fig, ax[1], im_junction, nuclei_orientation)
        else:
            Plotter._add_cell_orientation(fig, ax, im_junction, cell_orientation)

        # plot major and minor axis
        for index, row in collection.get_properties_by_img_name(img_name).iterrows():
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

        # set title and ax limits
        if inst_nuclei_mask is not None:
            add_title(
                ax[0],
                "cell shape orientation",
                im_junction.data,
                self.params.show_graphics_axis,
            )
            add_title(
                ax[1],
                "nuclei shape orientation",
                im_junction.data,
                self.params.show_graphics_axis,
            )
            axes = [ax[0], ax[1]]
        else:
            add_title(
                ax,
                "cell shape orientation",
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
        if self.params.plot_scalebar:
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
        """Plot all features of all images in the collection

        Args:
            collection:
                the collection to plot
            close:
                whether to close the figure after saving

        """
        for key in collection.img_dict.keys():
            img = collection.get_image_by_img_name(key)
            r_params = collection.get_runtime_params_by_img_name(key)

            if self.params.plot_polarity and img.has_nuclei() and img.has_organelle():
                self.plot_organelle_polarity(collection, key, close)
                if img.has_nuclei():
                    self.plot_nuc_displacement_orientation(collection, key, close)

            if self.params.plot_marker and img.has_marker():
                self.plot_marker_expression(collection, key, close)
                self.plot_marker_polarity(collection, key, close)
                if img.has_nuclei():
                    self.plot_marker_nucleus_orientation(collection, key, close)

                if self.params.plot_ratio_method:
                    self.plot_marker_cue_intensity_ratio(collection, key, close)

            if self.params.plot_junctions and img.has_junction():
                self.plot_junction_polarity(collection, key, close)
                self.plot_corners(collection, key, close)

            if self.params.plot_orientation:
                self.plot_eccentricity(collection, key, close)

            if self.params.plot_ratio_method:
                self.plot_junction_cue_intensity_ratio(collection, key, close)

            if self.params.plot_cyclic_orientation:
                self.plot_orientation(collection, key, close)

            if self.params.plot_foi:
                if r_params.extract_group_features:
                    self.plot_foi(collection, key, close)

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
            alpha=0.75,
        )

        # colorbar
        add_colorbar(fig, cax_1, ax, yticks, "shape orientation (degree)")

    @staticmethod
    def _add_cell_orientation(
            fig,
            ax,
            im_junction: BioMedicalChannel,
            cell_orientation: BioMedicalInstanceSegmentationMask,
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
            alpha=0.75,
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
    def _add_nuclei_eccentricity(
            fig,
            ax,
            im_junction: BioMedicalChannel,
            nuclei_eccentricity: BioMedicalInstanceSegmentationMask,
    ):
        v_min = 0.0
        v_max = 1.0
        yticks = [0.0, 0.5, 1.0]

        ax.imshow(im_junction.data, cmap=plt.cm.gray, alpha=1.0)

        # show nuclei eccentricity everywhere but background label
        cax_1 = ax.imshow(
            nuclei_eccentricity.mask_background().data,
            cmap=plt.cm.bwr,
            vmin=v_min,
            vmax=v_max,
            alpha=0.5,
        )

        # colorbar
        add_colorbar(fig, cax_1, ax, yticks, "eccentricity")

    @staticmethod
    def _add_cell_eccentricity(
            fig,
            ax,
            im_junction: BioMedicalChannel,
            cell_eccentricity: BioMedicalInstanceSegmentationMask,
    ):
        v_min = 0.0
        v_max = 1.0
        yticks = [0.0, 0.5, 1.0]

        ax.imshow(im_junction.data, cmap=plt.cm.gray, alpha=1.0)

        # show cell_eccentricity everywhere but background label
        cax_0 = ax.imshow(
            cell_eccentricity.mask_background().data,
            cmap=plt.cm.bwr,
            vmin=v_min,
            vmax=v_max,
            alpha=0.5,
        )

        # colorbar
        add_colorbar(fig, cax_0, ax, yticks, "eccentricity")

    @staticmethod
    def _add_single_cell_eccentricity_axis(
            ax,
            y0,
            x0,
            orientation,
            major_axis_length,
            minor_axis_length,
            eccentricity,
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

        ax.plot((y1_ma, y2_ma), (x1_ma, x2_ma), "--w", linewidth=0.5)
        ax.plot((y1_mi, y2_mi), (x1_mi, x2_mi), "--w", linewidth=0.5)
        ax.plot(y0, x0, ".b", markersize=markersize)
        ax.text(
            y0, x0, str(np.round(eccentricity, 2)), color=font_color, fontsize=fontsize
        )
