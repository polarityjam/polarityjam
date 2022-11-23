import numpy as np
from hashlib import sha1

from polarityjam.compute.moran import run_morans
from polarityjam.compute.neighborhood import k_neighbor_dif
from polarityjam.controller.collector import PropertyCollector, SingleCellPropertyCollector, SingleCellMaskCollector
from polarityjam.model.masks import MasksCollection
from polarityjam.model.parameter import RuntimeParameter
from polarityjam.polarityjam_logging import get_logger
from polarityjam.utils.rag import orientation_graph_nf, remove_islands


class Extractor:
    def __init__(self, params: RuntimeParameter):
        self.params = params
        self.collector = PropertyCollector()

    def threshold(self, single_cell_mask, single_nucleus_mask=None, single_organelle_mask=None):
        """Thresholds given single_cell_mask. Returns True if falls under threshold."""

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
    def get_image_marker(img, img_params):
        """Gets the image of the marker channel specified in the img_params."""
        if img_params.channel_expression_marker >= 0:
            get_logger().info("Marker channel at position: %s" % str(img_params.channel_expression_marker))
            return img[:, :, img_params.channel_expression_marker]
        return None

    @staticmethod
    def get_image_junction(img, img_params):
        """Gets the image of the junction channel specified in the img_params."""
        if img_params.channel_junction >= 0:
            get_logger().info("Junction channel at position: %s" % str(img_params.channel_junction))
            return img[:, :, img_params.channel_junction]
        return None

    @staticmethod
    def get_image_nucleus(img, img_params):
        """Gets the image of the nucleus channel specified in the img_params."""
        if img_params.channel_nucleus >= 0:
            get_logger().info("Nucleus channel at position: %s" % str(img_params.channel_nucleus))
            return img[:, :, img_params.channel_nucleus]
        return None

    @staticmethod
    def get_image_organelle(img, img_params):
        """Gets the image of the organelle channel specified in the img_params."""
        if img_params.channel_organelle >= 0:
            get_logger().info("Organelle channel at position: %s" % str(img_params.channel_organelle))
            return img[:, :, img_params.channel_organelle]
        return None

    @staticmethod
    def get_image_hash(img):
        return sha1(img.copy(order='C')).hexdigest()

    def extract(self, img, img_params, cells_mask, filename, output_path, collection):
        """ Extracts the features from an input image."""

        img_marker = self.get_image_marker(img, img_params)
        img_junction = self.get_image_junction(img, img_params)
        img_nucleus = self.get_image_nucleus(img, img_params)
        img_organelle = self.get_image_organelle(img, img_params)
        img_hash = self.get_image_hash(img)

        masks = MasksCollection(cells_mask)

        # initialize graph - no features associated with nodes
        rag = orientation_graph_nf(masks.cell_mask)
        list_of_islands = masks.set_cell_mask_rem_island(rag)
        rag = remove_islands(rag, list_of_islands)

        get_logger().info("Number of RAG nodes: %s " % len(list(rag.nodes)))

        if img_params.channel_nucleus >= 0:
            masks.set_nuclei_mask(img_nucleus)

        if img_params.channel_organelle >= 0:
            masks.set_organelle_mask(img_organelle)

        excluded = 0
        # iterate through each unique segmented cell
        for connected_component_label in np.unique(masks.cell_mask_rem_island):

            # ignore background
            if connected_component_label == 0:
                continue

            # get single cell masks
            sc_masks = SingleCellMaskCollector().calc_sc_masks(
                masks, connected_component_label, img_junction, self.params.membrane_thickness
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

            self.collector.collect_sc_props(sc_props_collection, collection, filename, img_hash,
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
        get_logger().info("Leftover cells: %s" % str(len(np.unique(masks.cell_mask)) - excluded))

        # morans I analysis based on FOI
        morans_i = run_morans(rag, self.params.feature_of_interest)
        num_cells = len(np.unique(masks.cell_mask_rem_island))
        self.collector.collect_group_statistic(collection, morans_i, num_cells)

        # neighborhood feature analysis based on FOI
        neighborhood_props_list = k_neighbor_dif(rag, self.params.feature_of_interest)
        self.collector.collect_neighborhood_props(collection, neighborhood_props_list)

        # mark the beginning of a new image that is potentially extracted
        self.collector.set_reset_index(collection)
        self.collector.add_out_path(collection, filename, output_path)
        self.collector.add_img(collection, filename, img_nucleus, img_junction, img_marker)
        self.collector.add_masks(collection, filename, masks)

        return collection
