import os
from hashlib import sha1
from pathlib import Path
from typing import Optional, Union

import numpy as np

from polarityjam import PropertiesCollection
from polarityjam.compute.neighborhood import k_neighbor_dif
from polarityjam.controller.collector import PropertyCollector, SingleCellPropertyCollector, SingleCellMaskCollector, \
    GroupPropertyCollector
from polarityjam.model.image import BioMedicalImage
from polarityjam.model.masks import MasksCollection, BioMedicalInstanceSegmentationMask, BioMedicalInstanceSegmentation, \
    MasksCollection_, BioMedicalInteriorMask
from polarityjam.model.moran import run_morans
from polarityjam.model.parameter import RuntimeParameter, ImageParameter
from polarityjam.polarityjam_logging import get_logger


class Extractor:
    def __init__(self, params: RuntimeParameter):
        self.params = params
        self.collector = PropertyCollector()

    def threshold_(self, sc_masks):
        if sc_masks.sc_mask.is_count_below_threshold(self.params.min_cell_size):
            return True
        if sc_masks.sc_nucleus_mask is not None:
            if sc_masks.sc_nucleus_mask.is_count_below_threshold(self.params.min_nucleus_size):
                return True
        if sc_masks.sc_organelle_mask is not None:
            if sc_masks.sc_organelle_mask.is_count_below_threshold(self.params.min_organelle_size):
                return True
        return False

    def threshold(
            self, single_cell_mask: np.ndarray,
            single_nucleus_mask: np.ndarray = None,
            single_organelle_mask: np.ndarray = None
    ) -> bool:
        """Thresholds given single_cell_mask. Returns True if falls under threshold.

        Args:
            single_cell_mask:
                The single cell mask to threshold.
            single_nucleus_mask:
                The single nucleus mask to threshold. If None, it will be ignored.
            single_organelle_mask:
                The single organelle mask to threshold. If None, it will be ignored.

        Returns:
            True if the single cell in the mask falls under the threshold.

        """

        # remove small cells
        if len(single_cell_mask[single_cell_mask == 1]) < self.params.min_cell_size:
            return True

        if single_nucleus_mask is not None:
            if len(single_nucleus_mask[single_nucleus_mask == 1]) < self.params.min_nucleus_size:
                return True

        if single_organelle_mask is not None:
            if len(single_organelle_mask[single_organelle_mask == 1]) < self.params.min_organelle_size:
                return True

        return False

    @staticmethod
    def get_image_marker(img: np.ndarray, img_params: ImageParameter) -> Optional[np.ndarray]:
        """Gets the image of the marker channel specified in the img_params.

        Args:
            img:
                The image to get the marker channel from.
            img_params:
                The image parameters specifying the channel position.

        Returns:
            The np.ndarray of the marker channel.

        """
        if img_params.channel_expression_marker >= 0:
            get_logger().info("Marker channel at position: %s" % str(img_params.channel_expression_marker))
            return img[:, :, img_params.channel_expression_marker]
        return None

    @staticmethod
    def get_image_junction(img: np.ndarray, img_params: ImageParameter) -> Optional[np.ndarray]:
        """Gets the image of the junction channel specified in the img_params.

        Args:
            img:
                The image to get the junction channel from.
            img_params:
                The image parameters specifying the channel position.

        Returns:
            The np.ndarray of the junction channel.

        """
        if img_params.channel_junction >= 0:
            get_logger().info("Junction channel at position: %s" % str(img_params.channel_junction))
            return img[:, :, img_params.channel_junction]
        return None

    @staticmethod
    def get_image_nucleus(img: np.ndarray, img_params: ImageParameter) -> Optional[np.ndarray]:
        """Gets the image of the nucleus channel specified in the img_params.

        Args:
            img:
                The image to get the nucleus channel from.
            img_params:
                The image parameters specifying the channel position.

        Returns:
            The np.ndarray of the nucleus channel.

        """
        if img_params.channel_nucleus >= 0:
            get_logger().info("Nucleus channel at position: %s" % str(img_params.channel_nucleus))
            return img[:, :, img_params.channel_nucleus]
        return None

    @staticmethod
    def get_image_organelle(img: np.ndarray, img_params: ImageParameter) -> Optional[np.ndarray]:
        """Gets the image of the organelle channel specified in the img_params.

        Args:
            img:
                The image to get the organelle channel from.
            img_params:
                The image parameters specifying the channel position.

        Returns:
            The np.ndarray of the organelle channel.
        """
        if img_params.channel_organelle >= 0:
            get_logger().info("Organelle channel at position: %s" % str(img_params.channel_organelle))
            return img[:, :, img_params.channel_organelle]
        return None

    @staticmethod
    def get_image_hash(img: np.ndarray) -> str:
        """Returns the hash of the given image.

        Args:
            img:
                The image to get the hash from.

        Returns:
            The hash of the image.

        """
        return sha1(img.copy(order='C')).hexdigest()

    def extract(
            self,
            img: np.ndarray,
            img_params: ImageParameter,
            cells_mask: np.ndarray,
            filename_prefix: str,
            output_path: Union[Path, str],
            collection: PropertiesCollection
    ) -> PropertiesCollection:
        """Extracts features from an input image into a given collection.

        Args:
            img:
                np.ndarray of the image to be processed.
            img_params:
                ImageParameter object containing the image parameters.
            cells_mask:
                np.ndarray of the cells mask.
            filename_prefix:
                Name prefix for the image used for all produced output.
            output_path:
                Path to the output directory.
            collection:
                PropertiesCollection object to which the extracted features will be added.

        Returns:
            PropertiesCollection object containing the extracted features.

        """
        filename_prefix, _ = os.path.splitext(os.path.basename(filename_prefix))
        ################################### new code ###################################
        bio_med_segmentation_mask = BioMedicalInstanceSegmentationMask(cells_mask)  # todo: rename to segmentation_mask
        bio_med_segmentation = BioMedicalInstanceSegmentation(bio_med_segmentation_mask)

        bio_med_image = BioMedicalImage(img, img_params, segmentation=bio_med_segmentation)

        nuclei_mask_seg = None
        if img_params.channel_nucleus >= 0:
            nuclei_mask_seg = BioMedicalInteriorMask(bio_med_image.nucleus.channel).to_instance_segmentation(
                bio_med_segmentation.segmentation_mask_connected)

        organelle_mask_seg = None
        if img_params.channel_organelle >= 0:
            organelle_mask_seg = BioMedicalInteriorMask(bio_med_image.organelle.channel).to_instance_segmentation(
                bio_med_segmentation.segmentation_mask_connected)

        mask_collection = MasksCollection_(
            bio_med_segmentation_mask, bio_med_segmentation.segmentation_mask_connected, nuclei_mask_seg,
            organelle_mask_seg
        )

        excluded = 0
        # iterate through each unique segmented cell
        for connected_component_label in np.unique(bio_med_segmentation.segmentation_mask_connected.mask):

            # ignore background
            if connected_component_label == 0:
                continue

            sc_masks = SingleCellMaskCollector.calc_sc_masks_(
                bio_med_image, connected_component_label, self.params.membrane_thickness, nuclei_mask_seg,
                organelle_mask_seg
            )

            # threshold
            if self.threshold_(sc_masks):
                get_logger().info(
                    "Cell \"%s\" falls under threshold! Removed from RAG..." % connected_component_label)
                excluded += 1
                # remove a cell from the RAG
                bio_med_segmentation.remove_component_label(connected_component_label)
                continue

            sc_props_collection = SingleCellPropertyCollector(self.params).calc_sc_props_(
                sc_masks, bio_med_image
            )

            PropertyCollector.collect_sc_props(sc_props_collection, collection, filename_prefix, bio_med_image.img_hash,
                                               connected_component_label)

            # append feature of interest to the RAG node for being able to do further analysis
            foi_val = PropertyCollector.get_foi(collection, self.params.feature_of_interest)

            bio_med_segmentation.set_feature_of_interest(connected_component_label, self.params.feature_of_interest,
                                                         foi_val)

        num_cells = len(np.unique(bio_med_segmentation.segmentation_mask_connected.mask)) - excluded
        get_logger().info("Excluded cells: %s" % str(excluded))
        get_logger().info("Leftover cells: %s" % str(num_cells))

        # morans I analysis based on FOI
        morans_i = GroupPropertyCollector.calc_moran(bio_med_segmentation, self.params.feature_of_interest)
        PropertyCollector.collect_group_statistic(collection, morans_i, num_cells)

        # neighborhood feature analysis based on FOI
        neighborhood_props_list = GroupPropertyCollector.calc_neighborhood(bio_med_segmentation,
                                                                           self.params.feature_of_interest)
        PropertyCollector.collect_neighborhood_props(collection, neighborhood_props_list)

        # mark the beginning of a new image that is potentially extracted
        PropertyCollector.set_reset_index(collection)
        PropertyCollector.add_out_path(collection, filename_prefix, output_path)
        PropertyCollector.add_foi(collection, filename_prefix, self.params.feature_of_interest)
        PropertyCollector.add_image_params(collection, filename_prefix, img_params)
        PropertyCollector.add_img_(collection, filename_prefix, bio_med_image)
        PropertyCollector.add_masks(collection, filename_prefix, mask_collection)

        return collection

        ################################### old code ###################################
        img_marker = self.get_image_marker(img, img_params)
        img_junction = self.get_image_junction(img, img_params)
        img_nucleus = self.get_image_nucleus(img, img_params)
        img_organelle = self.get_image_organelle(img, img_params)
        img_hash = self.get_image_hash(img)

        mask_collection = MasksCollection(cells_mask)

        rag, list_of_islands = mask_collection.set_cell_mask_connected()

        if img_params.channel_nucleus >= 0:
            mask_collection.set_nuclei_mask(img_nucleus)

        if img_params.channel_organelle >= 0:
            mask_collection.set_organelle_mask(img_organelle)

        excluded = 0
        # iterate through each unique segmented cell
        for connected_component_label in np.unique(mask_collection.cell_mask_connected):

            # ignore background
            if connected_component_label == 0:
                continue

            # get single cell masks
            sc_masks = SingleCellMaskCollector().calc_sc_masks(
                mask_collection, connected_component_label, img_junction, self.params.membrane_thickness
            )

            # threshold
            if self.threshold(
                    sc_masks.sc_mask,
                    single_nucleus_mask=sc_masks.sc_nucleus_mask,
                    single_organelle_mask=sc_masks.sc_organelle_mask
            ):
                get_logger().info("Cell \"%s\" falls under threshold! Removed from RAG..." % connected_component_label)
                excluded += 1
                # remove a cell from the RAG
                rag.remove_node(connected_component_label)
                continue

            sc_props_collection = SingleCellPropertyCollector(self.params).calc_sc_props(
                sc_masks, img_marker, img_junction
            )

            self.collector.collect_sc_props(sc_props_collection, collection, filename_prefix, img_hash,
                                            connected_component_label)

            # append feature of interest to the RAG node for being able to do further analysis
            foi_val = self.collector.get_foi(collection, self.params.feature_of_interest)
            rag.nodes[connected_component_label.astype('int')][self.params.feature_of_interest] = foi_val

            get_logger().info(
                " ".join(
                    str(x) for x in ["Cell %s - feature \"%s\": %s" % (
                        connected_component_label, self.params.feature_of_interest, foi_val
                    )]
                )
            )

        get_logger().info("Excluded cells: %s" % str(excluded))
        get_logger().info("Leftover cells: %s" % str(len(np.unique(mask_collection.cell_mask)) - excluded))

        # morans I analysis based on FOI
        morans_i = run_morans(rag, self.params.feature_of_interest)
        num_cells = len(np.unique(mask_collection.cell_mask_connected))
        self.collector.collect_group_statistic(collection, morans_i, num_cells)

        # neighborhood feature analysis based on FOI
        neighborhood_props_list = k_neighbor_dif(rag, self.params.feature_of_interest)
        self.collector.collect_neighborhood_props(collection, neighborhood_props_list)

        # mark the beginning of a new image that is potentially extracted
        self.collector.set_reset_index(collection)
        self.collector.add_out_path(collection, filename_prefix, output_path)
        self.collector.add_foi(collection, filename_prefix, self.params.feature_of_interest)
        self.collector.add_image_params(collection, filename_prefix, img_params)
        self.collector.add_img(collection, filename_prefix, img_nucleus, img_junction, img_marker)
        self.collector.add_masks(collection, filename_prefix, mask_collection)

        return collection
