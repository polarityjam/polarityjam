from typing import List, Tuple, Callable

import numpy as np
import skimage.filters
from scipy import ndimage as ndi
from skimage.future.graph import RAG

from polarityjam.polarityjam_logging import get_logger
from polarityjam.utils.rag import orientation_graph_nf, remove_islands


class SingleCellMasksCollection:
    """Collection of single cell masks."""

    def __init__(
            self,
            connected_component_label: int,
            sc_mask: np.ndarray,
            sc_nucleus_mask: np.ndarray,
            sc_organelle_mask: np.ndarray,
            sc_membrane_mask: np.ndarray,
            sc_cytosol_mask: np.ndarray,
            sc_junction_protein_mask: np.ndarray,
    ):
        self.connected_component_label = connected_component_label
        self.sc_mask = sc_mask
        self.sc_nucleus_mask = sc_nucleus_mask
        self.sc_organelle_mask = sc_organelle_mask
        self.sc_membrane_mask = sc_membrane_mask
        self.sc_cytosol_mask = sc_cytosol_mask
        self.sc_junction_protein_area_mask = sc_junction_protein_mask  # todo: rename


class MasksCollection:
    """Collection of masks."""

    def __init__(self, cell_mask: np.ndarray):
        self.cell_mask = cell_mask
        self.cell_mask_connected = None
        self.nuclei_mask = None
        self.organelle_mask = None

    def set_cell_mask_connected(self) -> Tuple[RAG, List[int]]:
        """Remove unconnected cells given a neighborhood graph (Cells without neighbours).

        Returns:
            Tuple of connected component neighborhood graph and unconnected component labels. (Islands)

        """
        neighborhood_graph = orientation_graph_nf(self.cell_mask)

        # Get list of islands - nodes with no neighbours and remove them
        list_of_islands = []
        for nodes in neighborhood_graph.nodes:
            if len(list(neighborhood_graph.neighbors(nodes))) == 0:
                list_of_islands.append(nodes)

        list_of_islands = np.unique(list_of_islands)

        mask = self.cell_mask

        # remove islands from mask
        for elemet in list_of_islands:
            mask[:, :][mask[:, :] == elemet] = 0

        self.cell_mask_connected = mask

        # remove islands from graph
        neighborhood_graph = remove_islands(neighborhood_graph, list_of_islands)

        get_logger().info("Removed number of islands: %s" % len(list_of_islands))
        get_logger().info("Number of RAG nodes: %s " % len(list(neighborhood_graph.nodes)))

        return neighborhood_graph, list_of_islands

    def set_nuclei_mask(self, img_nuclei: np.ndarray) -> np.ndarray:
        """Sets the nuclei mask.

        Args:
            img_nuclei:
               The nuclei image.

        Returns:
            The nuclei mask.
        """
        img_nuclei_blur = ndi.gaussian_filter(img_nuclei, sigma=3)
        nuclei_mask = np.where(img_nuclei_blur > skimage.filters.threshold_otsu(img_nuclei_blur), True, False)
        if self.cell_mask_connected is not None:
            nuclei_mask = nuclei_mask * self.cell_mask_connected
        self.nuclei_mask = nuclei_mask

        return nuclei_mask

    def set_organelle_mask(self, img_organelle: np.ndarray) -> np.ndarray:
        """Set the organelle mask.

        Args:
            img_organelle:
                The organelle image.

        Returns:
            The organelle mask.

        """
        img_organelle_blur = ndi.gaussian_filter(img_organelle, sigma=3)
        organelle_mask_o = np.where(
            img_organelle_blur > skimage.filters.threshold_otsu(img_organelle_blur), True, False
        )
        organelle_mask = organelle_mask_o * self.cell_mask_connected
        self.organelle_mask = organelle_mask

        return organelle_mask


class BioMedicalMask:
    def __init__(self, mask: np.ndarray):
        self.mask = mask

    def get_outline_from_mask(self, width: int = 1):
        """Computes outline for a single cell mask.

        Args:
            width:
                The width of the outline.

        Returns:
            The outline mask.

        """

        sc_mask = self.mask.astype(bool)
        dilated_mask = ndi.binary_dilation(sc_mask, iterations=width)
        eroded_mask = ndi.binary_erosion(sc_mask, iterations=width)
        outline_mask = np.logical_xor(dilated_mask, eroded_mask)

        return BioMedicalMask(outline_mask)

    def filter(self, overlay: object, operator: Callable):
        if not isinstance(overlay, BioMedicalMask):
            return NotImplemented

        return BioMedicalMask(operator(overlay.mask.astype(bool), self.mask.astype(bool)))

    def is_count_below_threshold(self, pixel_value: int) -> bool:
        if len(self.mask[self.mask == 1]) < pixel_value:
            return True


class BioMedicalInstanceSegmentationMask(BioMedicalMask):
    def __init__(self, mask: np.ndarray):
        super().__init__(mask)

    def get_single_cell_maks(self, connected_component_label: int) -> BioMedicalMask:
        """Gets the single cell mask given its label.

        Args:
            connected_component_label:
                The connected component label.

        Returns:
            The single cell mask.
        """
        return BioMedicalMask(
            np.where(self.mask == connected_component_label, 1, 0)
        )  # convert connected_component_label to 1


class BioMedicalInteriorMask(BioMedicalMask):
    def __init__(self, img_channel: np.ndarray):
        mask = self.get_interior_mask(img_channel)

        super().__init__(mask)

    def get_interior_mask(self, img_channel: np.ndarray) -> np.ndarray:
        img_channel_blur = ndi.gaussian_filter(img_channel, sigma=3)
        interior_mask = np.where(img_channel_blur > skimage.filters.threshold_otsu(img_channel_blur), True, False)

        return interior_mask

    def to_instance_segmentation(self,
                                 connected_component_mask: BioMedicalInstanceSegmentationMask) -> BioMedicalInstanceSegmentationMask:
        """Filters the nuclei mask.

        Args:
            connected_component_mask:
                The connected component mask.
        Returns:

        """
        return BioMedicalInstanceSegmentationMask(self.mask * connected_component_mask.mask)


class BioMedicalInstanceSegmentation:
    def __init__(self, segmentation_mask: BioMedicalInstanceSegmentationMask):
        self.segmentation_mask = segmentation_mask
        self.neighborhood_graph = orientation_graph_nf(self.segmentation_mask.mask)
        self.segmentation_mask_connected, self.list_of_islands = self.get_cell_mask_connected()

    def remove_component_label(self, connected_component_label):
        self.neighborhood_graph.remove_node(connected_component_label)
        # todo: should we remove the connected component from the segmentation mask?

    # todo: set for the whole image and not for each cell instance?
    def set_feature_of_interest(self, connected_component_label, feature_of_interest_name, feature_of_interest_val):
        self.neighborhood_graph.nodes[connected_component_label.astype('int')][
            feature_of_interest_name] = feature_of_interest_val

        get_logger().info(
            " ".join(
                str(x) for x in ["Cell %s - feature \"%s\": %s" % (
                    connected_component_label, feature_of_interest_name, feature_of_interest_val
                )]
            )
        )

    def get_cell_mask_connected(self) -> Tuple[BioMedicalInstanceSegmentationMask, List[int]]:
        """Remove unconnected cells from the mask (Cells without neighbours).

        Returns:
            List of all unconnected component labels. (Islands)

        """

        # Get list of islands - nodes with no neighbours and remove them
        list_of_islands = []
        for nodes in self.neighborhood_graph.nodes:
            if len(list(self.neighborhood_graph.neighbors(nodes))) == 0:
                list_of_islands.append(nodes)

        list_of_islands = np.unique(list_of_islands)

        connected_component_mask_np = np.copy(self.segmentation_mask.mask)

        # remove islands from mask
        for elemet in list_of_islands:
            connected_component_mask_np[:, :][connected_component_mask_np[:, :] == elemet] = 0

        # remove islands from graph
        self.neighborhood_graph = remove_islands(self.neighborhood_graph, list_of_islands)

        get_logger().info("Removed number of islands: %s" % len(list_of_islands))
        get_logger().info("Number of RAG nodes: %s " % len(list(self.neighborhood_graph.nodes)))

        connected_component_mask = BioMedicalInstanceSegmentationMask(connected_component_mask_np)

        return connected_component_mask, list_of_islands


def get_single_cell_mask(cells_mask: np.ndarray, connected_component_label: int) -> np.ndarray:
    """Gets the single cell mask from a cells mask given its label.

    Args:
        cells_mask:
            The cells mask.
        connected_component_label:
            The connected component label.

    Returns:
        The single cell mask.
    """
    return np.where(cells_mask == connected_component_label, 1, 0)  # convert connected_component_label to 1


def get_single_cell_nuc_mask(nuclei_mask: np.ndarray, cell_mask: np.ndarray,
                             connected_component_label: int) -> np.ndarray:
    """Gets the single nuclei mask from a cells mask given its label.

    Args:
        nuclei_mask:
            The nuclei mask.
        cell_mask:
            The cells mask.
        connected_component_label:
            The connected component label.

    Returns:
        The single nuclei mask.

    """
    single_cell_mask = get_single_cell_mask(cell_mask, connected_component_label)
    return single_cell_mask * nuclei_mask


def get_outline_from_mask(sc_mask: np.ndarray, width: int = 1) -> np.ndarray:
    """Computes outline for a single cell mask.

    Args:
        sc_mask:
            The single cell mask.
        width:
            The width of the outline.

    Returns:
        The outline mask.

    """

    sc_mask = sc_mask.astype(bool)
    dilated_mask = ndi.binary_dilation(sc_mask, iterations=width)
    eroded_mask = ndi.binary_erosion(sc_mask, iterations=width)
    outline_mask = np.logical_xor(dilated_mask, eroded_mask)

    return outline_mask


class SingleCellMasksCollection_:
    def __init__(
            self,
            connected_component_label: int,
            sc_mask: BioMedicalMask,
            sc_nucleus_mask: BioMedicalMask,
            sc_organelle_mask: BioMedicalMask,
            sc_membrane_mask: BioMedicalMask,
            sc_cytosol_mask: BioMedicalMask,
            sc_junction_protein_mask: BioMedicalMask,
    ):
        self.connected_component_label = connected_component_label
        self.sc_mask = sc_mask
        self.sc_nucleus_mask = sc_nucleus_mask
        self.sc_organelle_mask = sc_organelle_mask
        self.sc_membrane_mask = sc_membrane_mask
        self.sc_cytosol_mask = sc_cytosol_mask
        self.sc_junction_protein_area_mask = sc_junction_protein_mask


class MasksCollection_:
    def __init__(self, cell_mask: BioMedicalInstanceSegmentationMask,
                 cell_mask_connected: BioMedicalInstanceSegmentationMask,
                 nuclei_mask: BioMedicalInteriorMask, organelle_mask: BioMedicalInteriorMask):
        self.cell_mask = cell_mask
        self.cell_mask_connected = cell_mask_connected
        self.nuclei_mask = nuclei_mask
        self.organelle_mask = organelle_mask
