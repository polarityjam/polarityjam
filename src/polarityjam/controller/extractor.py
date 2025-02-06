"""Module that extracts features from an image."""
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from polarityjam import PropertiesCollection
from polarityjam.controller.collector import (
    GroupPropertyCollector,
    PropertyCollector,
    SingleCellPropertyCollector,
)
from polarityjam.model.image import BioMedicalImage, SingleCellImage
from polarityjam.model.masks import (
    BioMedicalInstanceSegmentation,
    BioMedicalInstanceSegmentationMask,
    BioMedicalJunctionSegmentation,
    BioMedicalMask,
)
from polarityjam.model.parameter import ImageParameter, RuntimeParameter
from polarityjam.polarityjam_logging import get_logger


class Extractor:
    """Extracts features from an image."""

    def __init__(self, params: RuntimeParameter):
        """Initialize the Extractor object."""
        self.params = params
        self.collector = PropertyCollector()

    def threshold_size(self, sc_image: SingleCellImage) -> bool:
        """Threshold a single cell image.

        Args:
            sc_image:
                SingleCellImage object containing the image information.

        Returns:
            True if any size count was below the threshold, False otherwise.

        """
        if sc_image.threshold_cell_size(self.params.min_cell_size):
            return True
        if sc_image.threshold_nucleus_size(self.params.min_nucleus_size):
            return True
        if sc_image.threshold_organelle_size(self.params.min_organelle_size):
            return True
        return False

    def extract_cell_features(
        self,
        collection: PropertiesCollection,
        bio_med_image: BioMedicalImage,
        filename_prefix: str,
    ):
        """Extract features from cells.

        Args:
            collection:
                PropertiesCollection object to which the extracted features will be added.
            bio_med_image:
                BioMedicalImage object containing the image information.
            filename_prefix:
                Name prefix for the image used for all produced output.

        """
        assert (
            bio_med_image.segmentation is not None
        ), "Segmentation needed to extract cell features!"

        # get all additional segmentation masks
        bio_med_image.segmentation.segmentation_mask_nuclei = self._get_nuclei_mask(
            bio_med_image
        )
        bio_med_image.segmentation.segmentation_mask_organelle = (
            self._get_organelle_mask(bio_med_image)
        )
        bio_med_image.segmentation.segmentation_mask_junction = self._get_junction_mask(
            bio_med_image
        )

        sc_image_list, threshold_cells = self._get_sc_images(bio_med_image)

        removed_islands = self._remove_threshold_cells(bio_med_image, threshold_cells)

        get_logger().info(
            "Number of additionally removed islands: %s" % len(removed_islands)
        )

        # remove sc images that have been excluded
        excluded_components = set(threshold_cells)
        excluded_components.update(list(bio_med_image.segmentation.island_list))

        sc_image_list = [
            i
            for i in sc_image_list
            if i.connected_component_label not in excluded_components
        ]

        get_logger().info(
            "Label of all excluded cells: %s"
            % str(",".join([str(i) for i in sorted(excluded_components)]))
        )
        get_logger().info(
            "Number of leftover cells: %s"
            % str(
                len(bio_med_image.segmentation.segmentation_mask_connected.get_labels())
            )
        )

        # calculate properties for each cell
        for sc_image in sc_image_list:
            sc_props_collection = SingleCellPropertyCollector.get_sc_props(
                sc_image, self.params
            )

            PropertyCollector.collect_sc_props(
                sc_props_collection,
                collection,
                filename_prefix,
                bio_med_image.img_hash,
                sc_image.connected_component_label,
                self.params,
            )

        if self.params.save_sc_image:
            # add sc_image to collection
            PropertyCollector.add_sc_imgs(collection, filename_prefix, sc_image_list)

    @staticmethod
    def _remove_threshold_cells(
        bio_med_image: BioMedicalImage, threshold_cells: List[int]
    ):
        assert (
            bio_med_image.segmentation is not None
        ), "Segmentation needed to extract cell features!"

        # remove threshold cells from segmentation
        islands_to_remove: List[int] = []
        conn_graph = bio_med_image.segmentation.connection_graph
        bio_med_image.segmentation.connection_graph = False

        for connected_component_label in threshold_cells[:-1]:
            bio_med_image.segmentation.remove_instance_label(connected_component_label)

        # remove islands after iteration
        bio_med_image.segmentation.connection_graph = conn_graph
        if len(threshold_cells) > 0:
            removed_islands_labels = bio_med_image.segmentation.remove_instance_label(
                threshold_cells[-1]
            )

            if removed_islands_labels:
                get_logger().info(
                    "Cell(s) with label(s) %s became isolated and have "
                    "additionally been removed from the adjacency graph!"
                    % ", ".join([str(x) for x in removed_islands_labels])
                )
                islands_to_remove.extend(removed_islands_labels)

        return islands_to_remove

    def _get_sc_images(
        self, bio_med_image: BioMedicalImage
    ) -> Tuple[List[SingleCellImage], List[int]]:
        sc_image_list = []
        threshold_cells = []
        assert (
            bio_med_image.segmentation is not None
        ), "Segmentation needed to extract cell features!"

        # iterate through each unique segmented cell
        for (
            cc_label
        ) in bio_med_image.segmentation.segmentation_mask_connected.get_labels():

            single_cell_image = bio_med_image.focus(
                cc_label,
                self.params.membrane_thickness,
                self.params.junction_threshold,
            )
            sc_image_list.append(single_cell_image)

            # threshold
            if self.threshold_size(single_cell_image):
                get_logger().info(
                    "Cell with label %s falls under threshold! Removed from analysis!"
                    % cc_label
                )
                threshold_cells.append(cc_label)
        return sc_image_list, threshold_cells

    @staticmethod
    def _get_organelle_mask(
        bio_med_image: BioMedicalImage,
    ) -> Optional[BioMedicalInstanceSegmentationMask]:
        organelle_mask_seg = None
        if bio_med_image.segmentation is not None:
            if bio_med_image.segmentation.segmentation_mask_organelle is not None:
                organelle_mask_seg = (
                    bio_med_image.segmentation.segmentation_mask_organelle
                )
            elif bio_med_image.has_organelle():
                assert (
                    bio_med_image.organelle is not None
                ), "Image has not organelle channel!"
                get_logger().info(
                    "Organelle channel found, but organelle mask not provided. "
                    "Retrieving mask via thresholding..."
                )
                organelle_mask_seg = BioMedicalMask.from_threshold_otsu(
                    bio_med_image.organelle.data
                ).overlay_instance_segmentation(
                    bio_med_image.segmentation.segmentation_mask_connected
                )  # convert to instance segmentation
        return organelle_mask_seg

    @staticmethod
    def _get_nuclei_mask(
        bio_med_image: BioMedicalImage,
    ) -> Optional[BioMedicalInstanceSegmentationMask]:
        nuclei_mask_seg = None
        if bio_med_image.segmentation is not None:
            if bio_med_image.segmentation.segmentation_mask_nuclei is not None:
                nuclei_mask_seg = bio_med_image.segmentation.segmentation_mask_nuclei
            elif bio_med_image.has_nuclei():
                assert (
                    bio_med_image.nucleus is not None
                ), "Image has not nucleus channel!"
                get_logger().info(
                    "Nuclei channel found, but nuclei mask not provided. "
                    "Retrieving mask via thresholding..."
                )
                nuclei_mask_seg = BioMedicalMask.from_threshold_otsu(
                    bio_med_image.nucleus.data
                ).overlay_instance_segmentation(
                    bio_med_image.segmentation.segmentation_mask_connected
                )  # convert to instance segmentation
        return nuclei_mask_seg

    @staticmethod
    def _get_junction_mask(
        bio_med_image: BioMedicalImage,
    ) -> Optional[BioMedicalJunctionSegmentation]:
        junction_mask_seg = None
        if bio_med_image.segmentation is not None:
            if bio_med_image.segmentation.segmentation_mask_junction is not None:
                junction_mask_seg = (
                    bio_med_image.segmentation.segmentation_mask_junction
                )
            elif bio_med_image.has_junction():
                assert (
                    bio_med_image.junction is not None
                ), "Image has no junction channel!"
                get_logger().info(
                    "Junction channel found, but junction mask not provided. "
                    "Mask will be created via thresholding..."
                )
                # this happens on single cell level, as we do not want to threshold the whole image
                return None

        return junction_mask_seg

    def extract(
        self,
        img: np.ndarray,
        img_params: ImageParameter,
        segmentation_mask: np.ndarray,
        filename_prefix: str,
        output_path: Union[Path, str],
        collection: PropertiesCollection,
        segmentation_mask_nuclei: Optional[
            Union[np.ndarray, BioMedicalInstanceSegmentationMask]
        ] = None,
        segmentation_mask_organelle: Optional[
            Union[np.ndarray, BioMedicalInstanceSegmentationMask]
        ] = None,
        segmentation_mask_junction: Optional[BioMedicalJunctionSegmentation] = None,
    ) -> PropertiesCollection:
        """Extract features from an input image into a given collection.

        Args:
            img:
                np.ndarray of the image to be processed.
            img_params:
                ImageParameter object containing the image parameters.
            segmentation_mask:
                np.ndarray of the cells mask.
            filename_prefix:
                Name prefix for the image used for all produced output.
            output_path:
                Path to the output directory.
            collection:
                PropertiesCollection object to which the extracted features will be added.
            segmentation_mask_nuclei:
                np.ndarray of the nuclei mask. Enhances feature quality. Optional.
            segmentation_mask_organelle:
                np.ndarray of the organelle mask. Enhances feature quality. Optional.
            segmentation_mask_junction:
                BioMedicalJunctionSegmentation. Enhances feature quality. Optional.

        Returns:
            PropertiesCollection object containing the extracted features.

        """
        get_logger().info("Extracting features for file %s..." % str(filename_prefix))
        filename_prefix, _ = os.path.splitext(os.path.basename(filename_prefix))

        get_logger().info("Prepare cell segmentation...")
        bio_med_segmentation_mask = self.prepare_segmentation(
            BioMedicalInstanceSegmentationMask(segmentation_mask)
        )

        if isinstance(segmentation_mask_nuclei, np.ndarray):
            get_logger().info("Prepare nuclei segmentation...")
            segmentation_mask_nuclei = self.prepare_segmentation(
                BioMedicalInstanceSegmentationMask(segmentation_mask_nuclei)
            )
        if isinstance(segmentation_mask_organelle, np.ndarray):
            get_logger().info("Prepare organelle segmentation...")
            segmentation_mask_organelle = self.prepare_segmentation(
                BioMedicalInstanceSegmentationMask(segmentation_mask_organelle)
            )

        bio_med_segmentation = BioMedicalInstanceSegmentation(
            bio_med_segmentation_mask,
            segmentation_mask_nuclei=segmentation_mask_nuclei,
            segmentation_mask_organelle=segmentation_mask_organelle,
            segmentation_mask_junction=segmentation_mask_junction,
            connection_graph=self.params.connection_graph,
        )

        get_logger().info(
            "Detected islands in the adjacency graph: %s"
            % ", ".join([str(x) for x in sorted(bio_med_segmentation.island_list)])
            if self.params.connection_graph
            else "Detected islands in the adjacency graph: Disabled!"
        )

        bio_med_image = BioMedicalImage(
            img, img_params, segmentation=bio_med_segmentation
        )

        # add all necessary information to the collection
        PropertyCollector.add_out_path(collection, filename_prefix, output_path)
        PropertyCollector.add_runtime_params(collection, filename_prefix, self.params)
        PropertyCollector.add_img(collection, filename_prefix, bio_med_image)

        self.extract_cell_features(collection, bio_med_image, filename_prefix)

        if self.params.extract_group_features:
            self.extract_group_features(
                collection, bio_med_segmentation, filename_prefix
            )

        # mark the beginning of a new image that is potentially extracted
        PropertyCollector.set_reset_index(collection)

        get_logger().info("Done feature extraction for file: %s" % str(filename_prefix))

        return collection

    def prepare_segmentation(
        self, inst_mask: BioMedicalInstanceSegmentationMask
    ) -> BioMedicalInstanceSegmentationMask:
        """Prepare a segmentation mask for further processing.

        Args:
            inst_mask:
                np.ndarray of the cellpose segmentation mask.

        Returns:
            np.ndarray of the prepared segmentation mask.

        """
        if self.params.clear_border:
            mask_clear_border = inst_mask.clear_border()
            number_of_cellpose_borders = len(inst_mask.get_labels()) - len(
                mask_clear_border.get_labels()
            )

            get_logger().info(
                "Removed number of border cells: %s" % number_of_cellpose_borders
            )
            inst_mask = mask_clear_border

        if self.params.remove_small_objects_size > 0:
            mask_rm_small_objects = inst_mask.remove_small_objects(
                self.params.remove_small_objects_size
            )
            number_of_cellpose_small_objects = len(inst_mask.get_labels()) - len(
                mask_rm_small_objects.get_labels()
            )
            inst_mask = mask_rm_small_objects

            get_logger().info(
                "Removed number of small objects: %s" % number_of_cellpose_small_objects
            )
        get_logger().info("Preparation done!")
        return inst_mask

    def extract_group_features(
        self,
        collection: PropertiesCollection,
        bio_med_segmentation: BioMedicalInstanceSegmentation,
        filename_prefix: str,
    ):
        """Extract features from a group of cells.

        Args:
            bio_med_segmentation:
                BioMedicalInstanceSegmentation object containing the segmentation of the cells.
            collection:
                PropertiesCollection object to which the extracted features will be added.
            filename_prefix:
                Name prefix for the image used for all produced output.

        """
        if len(collection) < 2:
            get_logger().warning(
                """Neighborhood analysis not possible.
                 Not enough cells to calculate group features!
                 Switch off neighborhood analysis (extract_group_features = False) or improve segmentation!"""
            )
            return
        foi_ds = collection.get_properties_by_img_name(filename_prefix)
        # catch whether the feature of interest is present
        if self.params.feature_of_interest not in foi_ds.columns:
            get_logger().warning(
                "Feature of interest not present in dataset."
                " Did you enable the respective feature class for extraction?"
            )
            return

        foi_vec = foi_ds[self.params.feature_of_interest].values
        bio_med_segmentation.set_feature_of_interest(
            self.params.feature_of_interest, foi_vec
        )

        # morans I analysis based on FOI
        morans_i = GroupPropertyCollector.calc_moran(
            bio_med_segmentation, self.params.feature_of_interest
        )
        PropertyCollector.collect_group_statistic(
            collection,
            morans_i,
            len(bio_med_segmentation.segmentation_mask_connected),
            self.params,
        )

        # neighborhood feature analysis based on FOI
        neighborhood_props_list = GroupPropertyCollector.calc_neighborhood(
            bio_med_segmentation, self.params.feature_of_interest
        )
        PropertyCollector.collect_neighborhood_props(
            collection, neighborhood_props_list, self.params
        )
