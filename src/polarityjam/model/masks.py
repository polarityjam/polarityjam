import numpy as np
import skimage.filters
from scipy import ndimage as ndi

from polarityjam.polarityjam_logging import get_logger


class SingleCellMasksCollection:
    def __init__(self, connected_component_label, sc_mask, sc_nucleus_mask, sc_organelle_mask, sc_membrane_mask,
                 sc_cytosol_mask, sc_junction_protein_mask):
        self.connected_component_label = connected_component_label
        self.sc_mask = sc_mask
        self.sc_nucleus_mask = sc_nucleus_mask
        self.sc_organelle_mask = sc_organelle_mask
        self.sc_membrane_mask = sc_membrane_mask
        self.sc_cytosol_mask = sc_cytosol_mask
        self.sc_junction_protein_area_mask = sc_junction_protein_mask  # todo: rename


class MasksCollection:
    def __init__(self, cell_mask):
        self.cell_mask = cell_mask
        self.cell_mask_rem_island = None
        self.nuclei_mask = None
        self.organelle_mask = None

    def set_cell_mask_rem_island(self, neighborhood_graph):
        """Remove unconnected cells (Cells without neighbours)."""
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

        self.cell_mask_rem_island = mask

        get_logger().info("Removed number of islands: %s" % len(list_of_islands))

        return list_of_islands

    def set_nuclei_mask(self, img_nuclei):
        """Sets the nuclei mask."""
        img_nuclei_blur = ndi.gaussian_filter(img_nuclei, sigma=3)
        nuclei_mask = np.where(img_nuclei_blur > skimage.filters.threshold_otsu(img_nuclei_blur), True, False)
        if self.cell_mask_rem_island is not None:
            nuclei_mask = nuclei_mask * self.cell_mask_rem_island
        self.nuclei_mask = nuclei_mask

        return nuclei_mask

    def set_organelle_mask(self, img_organelle):
        """Set the organelle mask."""
        img_organelle_blur = ndi.gaussian_filter(img_organelle, sigma=3)
        organelle_mask_o = np.where(
            img_organelle_blur > skimage.filters.threshold_otsu(img_organelle_blur), True, False
        )
        organelle_mask = organelle_mask_o * self.cell_mask_rem_island
        self.organelle_mask = organelle_mask

        return organelle_mask


def get_single_cell_mask(cell_mask, connected_component_label):
    """Gets the single cell mask from a mask where each cell has an increasing connected component value."""
    return np.where(cell_mask == connected_component_label, 1, 0)  # convert connected_component_label to 1


def get_outline_from_mask(mask, width=1):
    """Computes outline for a mask with a single label"""

    mask = mask.astype(bool)
    dilated_mask = ndi.binary_dilation(mask, iterations=width)
    eroded_mask = ndi.binary_erosion(mask, iterations=width)
    outline_mask = np.logical_xor(dilated_mask, eroded_mask)

    return outline_mask
