import os
from pathlib import Path
from typing import Union

import numpy as np

from polarityjam import PropertiesCollection
from polarityjam.controller.collector import (GroupPropertyCollector,
                                              PropertyCollector,
                                              SingleCellMaskCollector,
                                              SingleCellPropertyCollector)
from polarityjam.model.image import BioMedicalImage
from polarityjam.model.masks import (BioMedicalInstanceSegmentation,
                                     BioMedicalInstanceSegmentationMask,
                                     BioMedicalMask, SingleCellMasksCollection)
from polarityjam.model.parameter import ImageParameter, RuntimeParameter
from polarityjam.polarityjam_logging import get_logger


class Extractor:
    def __init__(self, params: RuntimeParameter):
        self.params = params
        self.collector = PropertyCollector()

    def threshold_size(self, sc_masks: SingleCellMasksCollection) -> bool:
        """Thresholds a collection of single cell mask based on cell size, nucleus size and organelle size.

        Args:
            sc_masks:
                The collection of single cell masks to threshold.

        Returns:
            True if the size count was below the threshold, False otherwise.

        """
        if sc_masks.sc_mask.is_count_below_threshold(self.params.min_cell_size):
            return True
        if sc_masks.sc_nucleus_mask is not None:
            if sc_masks.sc_nucleus_mask.is_count_below_threshold(
                self.params.min_nucleus_size
            ):
                return True
        if sc_masks.sc_organelle_mask is not None:
            if sc_masks.sc_organelle_mask.is_count_below_threshold(
                self.params.min_organelle_size
            ):
                return True
        return False

    def extract_cell_features(
        self, collection, bio_med_image, bio_med_segmentation, filename_prefix
    ):
        """Extracts features from cells

        Args:
            collection:
                PropertiesCollection object to which the extracted features will be added.
            bio_med_image:
                BioMedicalImage object containing the image information.
            bio_med_segmentation:
                BioMedicalInstanceSegmentation object containing the segmentation of the cells.
            filename_prefix:
                Name prefix for the image used for all produced output.

        """
        nuclei_mask_seg = None
        if bio_med_image.has_nuclei():
            nuclei_mask_seg = BioMedicalMask.from_threshold_otsu(
                bio_med_image.nucleus.data
            ).overlay_instance_segmentation(
                bio_med_segmentation.segmentation_mask_connected
            )

        organelle_mask_seg = None
        if bio_med_image.has_organelle():
            organelle_mask_seg = BioMedicalMask.from_threshold_otsu(
                bio_med_image.organelle.data
            ).overlay_instance_segmentation(
                bio_med_segmentation.segmentation_mask_connected
            )

        excluded = 0
        sc_masks_list = []
        # iterate through each unique segmented cell
        for (
            connected_component_label
        ) in bio_med_segmentation.segmentation_mask_connected.get_labels():

            sc_masks = SingleCellMaskCollector.calc_sc_masks(
                bio_med_image,
                connected_component_label,
                self.params.membrane_thickness,
                nuclei_mask_seg,
                organelle_mask_seg,
            )

            # threshold
            if self.threshold_size(sc_masks):
                get_logger().info(
                    'Cell "%s" falls under threshold! Removed from analysis!'
                    % connected_component_label
                )
                # remove a cell from the segmentation
                removed_islands_labels = bio_med_segmentation.remove_instance_label(
                    connected_component_label
                )

                if removed_islands_labels:
                    get_logger().info(
                        'Cell(s) "%s" became isolated and have additionally ben removed from the analysis!'
                        % ", ".join(removed_islands_labels)
                    )

                # remove instance from the organelle instance mask
                if bio_med_image.has_organelle():
                    organelle_mask_seg = organelle_mask_seg.remove_instance(
                        connected_component_label
                    )
                    # additionally remove the islands from the organelle mask
                    [
                        organelle_mask_seg.remove_instance(i)
                        for i in removed_islands_labels
                    ]

                # remove instance from the nucleus instance mask
                if bio_med_image.has_nuclei():
                    nuclei_mask_seg = nuclei_mask_seg.remove_instance(
                        connected_component_label
                    )
                    # additionally remove the islands from the nuclei mask
                    [nuclei_mask_seg.remove_instance(i) for i in removed_islands_labels]

                # exclude all islands from the analysis - e.g. keep only those that have not been removed from the RAG
                sc_masks_list = [
                    i
                    for i in sc_masks_list
                    if i.connected_component_label not in removed_islands_labels
                ]

                # counter for excluded cells
                excluded += 1 + len(removed_islands_labels)
                continue

            sc_masks_list.append(sc_masks)

        if bio_med_image.has_organelle():
            bio_med_image.organelle.add_mask("organelle_mask_seg", organelle_mask_seg)

        if bio_med_image.has_nuclei():
            bio_med_image.nucleus.add_mask("nuclei_mask_seg", nuclei_mask_seg)

        num_cells = len(bio_med_segmentation.segmentation_mask_connected) - excluded
        get_logger().info("Excluded cells: %s" % str(excluded))
        get_logger().info("Leftover cells: %s" % str(num_cells))

        # calculate properties for each cell
        for sc_masks in sc_masks_list:
            sc_props_collection = SingleCellPropertyCollector.calc_sc_props(
                sc_masks, bio_med_image, self.params
            )

            PropertyCollector.collect_sc_props(
                sc_props_collection,
                collection,
                filename_prefix,
                bio_med_image.img_hash,
                sc_masks.connected_component_label,
            )

    def extract(
        self,
        img: np.ndarray,
        img_params: ImageParameter,
        segmentation_mask: np.ndarray,
        filename_prefix: str,
        output_path: Union[Path, str],
        collection: PropertiesCollection,
    ) -> PropertiesCollection:
        """Extracts features from an input image into a given collection.

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

        Returns:
            PropertiesCollection object containing the extracted features.

        """
        filename_prefix, _ = os.path.splitext(os.path.basename(filename_prefix))

        bio_med_segmentation_mask = BioMedicalInstanceSegmentationMask(
            segmentation_mask
        )
        bio_med_segmentation = BioMedicalInstanceSegmentation(bio_med_segmentation_mask)

        bio_med_image = BioMedicalImage(
            img, img_params, segmentation=bio_med_segmentation
        )

        self.extract_cell_features(
            collection, bio_med_image, bio_med_segmentation, filename_prefix
        )

        if self.params.extract_group_features:
            self.extract_group_features(
                collection, bio_med_segmentation, filename_prefix
            )

        # mark the beginning of a new image that is potentially extracted
        PropertyCollector.set_reset_index(collection)
        PropertyCollector.add_out_path(collection, filename_prefix, output_path)
        PropertyCollector.add_runtime_params(collection, filename_prefix, self.params)
        PropertyCollector.add_img(collection, filename_prefix, bio_med_image)

        get_logger().info("Done feature extraction for file: %s" % str(filename_prefix))

        return collection

    def extract_group_features(
        self,
        collection: PropertiesCollection,
        bio_med_segmentation: BioMedicalInstanceSegmentation,
        filename_prefix: str,
    ):
        """Extracts features from a group of cells.

        Args:
            bio_med_segmentation:
                BioMedicalInstanceSegmentation object containing the segmentation of the cells.
            collection:
                PropertiesCollection object to which the extracted features will be added.
            filename_prefix:
                Name prefix for the image used for all produced output.

        """

        if len(collection) < 2:
            get_logger().warn(
                """Neighborhood analysis not possible.
                 Not enough cells to calculate group features!
                 Switch off neighborhood analysis (extract_group_features = False) or improve segmentation!"""
            )
            return

        foi_vec = collection.get_properties_by_img_name(filename_prefix)[
            self.params.feature_of_interest
        ].values
        bio_med_segmentation.set_feature_of_interest(
            self.params.feature_of_interest, foi_vec
        )

        # morans I analysis based on FOI
        morans_i = GroupPropertyCollector.calc_moran(
            bio_med_segmentation, self.params.feature_of_interest
        )
        PropertyCollector.collect_group_statistic(
            collection, morans_i, len(bio_med_segmentation.segmentation_mask_connected)
        )

        # neighborhood feature analysis based on FOI
        neighborhood_props_list = GroupPropertyCollector.calc_neighborhood(
            bio_med_segmentation, self.params.feature_of_interest
        )
        PropertyCollector.collect_neighborhood_props(
            collection, neighborhood_props_list
        )
