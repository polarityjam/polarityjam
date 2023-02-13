from __future__ import annotations

from typing import List, Tuple, Callable, Union, Any

import numpy as np
import skimage.filters
from scipy import ndimage as ndi
from skimage.future.graph import RAG

from polarityjam.polarityjam_logging import get_logger


class Mask:
    """Class representing a base mask."""

    def __init__(self, mask: np.ndarray):
        self.data = mask.astype(bool)

    @classmethod
    def from_threshold_otsu(cls, channel: np.ndarray, gaussian_filter=3) -> Mask:
        """Initializes a mask from a channel using Otsu's method."""
        if gaussian_filter is not None:
            img_channel_blur = ndi.gaussian_filter(channel, sigma=3)
        else:
            img_channel_blur = channel
        interior_mask = np.where(img_channel_blur > skimage.filters.threshold_otsu(img_channel_blur), True, False)

        return cls(interior_mask)

    @classmethod
    def empty(cls, shape: Tuple[int, int]) -> Mask:
        """Creates an empty mask of a given shape."""
        return cls(np.zeros(shape))

    def is_count_below_threshold(self, pixel_value: int) -> bool:
        """Checks if the number of pixels in the mask is below a given threshold.

        Args:
            pixel_value:
                The threshold value.

        Returns:
            True if the number of pixels is below the threshold, False otherwise.
        """
        if len(self.data[self.data == 1]) < pixel_value:
            return True
        return False


class BioMedicalMask(Mask):
    """Class representing a single boolean mask."""

    def __init__(self, mask: np.ndarray):
        super().__init__(mask)

    def invert(self) -> BioMedicalMask:
        """Inverts the mask."""
        return BioMedicalMask(np.invert(self.data))

    def operation(self, overlay: BioMedicalMask, operator: Callable) -> BioMedicalMask:
        """Performs an operation on the mask with another mask."""
        return BioMedicalMask(operator(overlay.data.astype(bool), self.data.astype(bool)))

    def get_outline_from_mask(self, width: int = 1) -> BioMedicalMask:
        """Computes outline for a single cell mask.

        Args:
            width:
                The width of the outline.

        Returns:
            The outline mask.

        """

        sc_mask = self.data.astype(bool)
        dilated_mask = ndi.binary_dilation(sc_mask, iterations=width)
        eroded_mask = ndi.binary_erosion(sc_mask, iterations=width)
        outline_mask = np.logical_xor(dilated_mask, eroded_mask)

        return BioMedicalMask(outline_mask)

    def overlay_instance_segmentation(
            self,
            connected_component_mask: BioMedicalInstanceSegmentationMask
    ) -> BioMedicalInstanceSegmentationMask:
        """Overlays an instance segmentation mask on the mask."""
        return BioMedicalInstanceSegmentationMask(self.data * connected_component_mask.data)

    def to_instance_mask(self, instance_label: int = 1) -> BioMedicalInstanceSegmentationMask:
        """Converts the mask to an instance segmentation mask."""
        return BioMedicalInstanceSegmentationMask(self.data * instance_label)

    def mask_background(
            self,
    ) -> BioMedicalMask:
        """Masks the background of the mask."""
        return BioMedicalMask(
            np.ma.masked_where(self.data == False, self.data)
        )

    def combine(self, other_mask: BioMedicalMask):
        return BioMedicalMask(np.logical_and(self.data, other_mask.data))


class BioMedicalInstanceSegmentationMask(Mask):
    """Class representing an instance segmentation mask."""

    def __init__(self, mask: np.ndarray, dtype: Union[str, Any] = int, background_label: Union[int, float] = 0):
        super().__init__(mask)
        self.background_label = background_label
        self.data = mask.astype(dtype)

    def remove_instance(self, instance_label: int) -> BioMedicalInstanceSegmentationMask:
        """Removes a single instance from the mask."""
        return BioMedicalInstanceSegmentationMask(
            np.where(self.data == instance_label, self.background_label, self.data),
            background_label=self.background_label
        )

    def mask_background(
            self, background_label: Union[int, float] = 0, mask_with: Union[
                np.ndarray, BioMedicalInstanceSegmentationMask] = None
    ) -> BioMedicalInstanceSegmentationMask:
        """Masks the background of the mask."""
        if mask_with is None:
            mask_with = self.data
        elif isinstance(mask_with, BioMedicalInstanceSegmentationMask):
            mask_with = mask_with.data

        return BioMedicalInstanceSegmentationMask(
            np.ma.masked_where(self.data == background_label, mask_with), dtype=float,
            background_label=self.background_label
        )

    def get_single_instance_maks(self, instance_label: int) -> BioMedicalMask:
        """Gets the single cell mask given its label.

        Args:
            instance_label:
                The connected component label.

        Returns:
            The single cell mask.
        """
        return BioMedicalMask(
            np.where(self.data == instance_label, True, False)
        )  # convert connected_component_label to True/False mask

    def __len__(self):
        return len(np.unique(self.data))

    def get_labels(self, exclude_background: bool = True) -> np.ndarray:
        """Gets the labels of the mask.

        Args:
            exclude_background:
                Whether to exclude the background label.

        Returns:
            The labels of the mask.

        """
        labels = np.unique(self.data)

        if exclude_background:
            labels = labels[labels != self.background_label]

        return labels

    def to_semantic_mask(self) -> BioMedicalMask:
        """Converts the mask to a semantic (boolean) mask."""
        return BioMedicalMask(self.data.astype(bool))

    def element_mult(
            self, mask: Union[BioMedicalInstanceSegmentationMask, np.ndarray]
    ) -> BioMedicalInstanceSegmentationMask:
        """Performs an element-wise multiplication of the mask with another mask.

        Args:
            mask:
                The mask to multiply with.

        Returns:
            The resulting mask.

        """
        if isinstance(mask, BioMedicalInstanceSegmentationMask):
            mask = mask.data
        return BioMedicalInstanceSegmentationMask(self.data * mask, dtype=float, background_label=self.background_label)

    def element_add(
            self, mask: Union[BioMedicalInstanceSegmentationMask, np.ndarray]
    ) -> BioMedicalInstanceSegmentationMask:
        """Performs an element-wise addition of the mask with another mask.

        Args:
            mask:
                The mask to add with.

        Returns:
            The resulting mask.

        """
        if isinstance(mask, BioMedicalInstanceSegmentationMask):
            mask = mask.data
        return BioMedicalInstanceSegmentationMask(self.data + mask, dtype=float, background_label=self.background_label)

    def scalar_mult(self, scalar: int) -> BioMedicalInstanceSegmentationMask:
        """Performs a scalar multiplication of the mask.

        Args:
            scalar:
                The scalar to multiply with.

        Returns:
            The resulting mask.

        """
        return BioMedicalInstanceSegmentationMask(self.data * scalar, dtype=float,
                                                  background_label=self.background_label)

    def scalar_add(self, scalar: int) -> BioMedicalInstanceSegmentationMask:
        """Performs a scalar addition of the mask.

        Args:
            scalar:
                The scalar to add with.

        Returns:
            The resulting mask.

        """
        return BioMedicalInstanceSegmentationMask(self.data + scalar, dtype=float,
                                                  background_label=self.background_label)

    def relabel(self, new_labels: Union[dict, np.ndarray],
                exclude_background: bool = True) -> BioMedicalInstanceSegmentationMask:
        """Reallocates the labels of the mask.

        Args:
            new_labels:
                A dictionary mapping old labels to new labels.
                Alternatively, a numpy array that assumes consecutive ordered labels.
            exclude_background:
                Whether to exclude the background label from the reallocation.

        Returns:
            New mask object with a reallocated mask.

        """
        l = len(self) - 1 if exclude_background else len(self)

        old_labels = np.unique(self.data)

        if exclude_background:
            old_labels = old_labels[old_labels != self.background_label]

        if isinstance(new_labels, dict):
            new_labels = np.array([new_labels[label] for label in old_labels])

        err_str = f"The number of new labels must match the number of old labels. Got length %s and length %s." % (
            len(new_labels), l
        )
        assert len(new_labels) == l, err_str

        new_mask = np.zeros_like(self.data).astype(new_labels.dtype)
        for old_label, new_label in zip(old_labels, new_labels):
            new_mask[self.data == old_label] = new_label

        return BioMedicalInstanceSegmentationMask(new_mask, dtype=new_mask.dtype,
                                                  background_label=self.background_label)


class BioMedicalInstanceSegmentation:
    """Class representing an entire instance segmentation."""

    def __init__(self, segmentation_mask: BioMedicalInstanceSegmentationMask):
        self.segmentation_mask = segmentation_mask
        self.neighborhood_graph = BioMedicalInstanceSegmentation.get_rag(self.segmentation_mask)
        self.segmentation_mask_connected = self.init_segmentation_mask(
            BioMedicalInstanceSegmentation.get_islands(self.neighborhood_graph))
        self.neighborhood_graph_connected = BioMedicalInstanceSegmentation.get_rag(self.segmentation_mask_connected)

    def update_graphs(self):
        self.neighborhood_graph = BioMedicalInstanceSegmentation.get_rag(self.segmentation_mask)
        self.neighborhood_graph_connected = BioMedicalInstanceSegmentation.get_rag(
            self.segmentation_mask_connected)  # could have islands
        self.segmentation_mask_connected, island_list = self.remove_islands()

        return island_list

    def init_segmentation_mask(self, islands: np.ndarray):
        connected_component_mask = BioMedicalInstanceSegmentationMask(np.copy(self.segmentation_mask.data))

        # remove islands from mask
        for elemet in islands:
            connected_component_mask = connected_component_mask.remove_instance(elemet)

        return connected_component_mask

    def remove_instance_label(self, instance_label):
        """Removes an instance label from the segmentation inplace.

        Args:
            instance_label:
                The instance label to remove.

        Returns:
            All cells that have been additionally removed because they were not connected to the main graph anymore.

        """
        self.neighborhood_graph.remove_node(instance_label)
        self.segmentation_mask = self.segmentation_mask.remove_instance(instance_label)
        self.segmentation_mask_connected = self.segmentation_mask_connected.remove_instance(
            instance_label)

        return self.update_graphs()

    def set_feature_of_interest(self, feature_of_interest_name: str, feature_of_interest_vec: np.ndarray):
        nodes = sorted(self.neighborhood_graph_connected.nodes)

        assert len(feature_of_interest_vec) == len(
            nodes), "The length of the feature of interest vector must match the number of nodes in the graph."

        for idx, node in enumerate(nodes):
            self.neighborhood_graph_connected.nodes[node][feature_of_interest_name] = feature_of_interest_vec[idx]

    def remove_islands(self) -> Tuple[BioMedicalInstanceSegmentationMask, List[int]]:
        """Remove unconnected cells from the mask (Cells without neighbours).

        Returns:
            A tuple containing the connected mask and a list of unconnected cells (Islands).

        """

        # Get list of islands - nodes with no neighbours and remove them
        list_of_islands = BioMedicalInstanceSegmentation.get_islands(self.neighborhood_graph_connected)

        connected_component_mask = BioMedicalInstanceSegmentationMask(np.copy(self.segmentation_mask_connected.data))

        # remove islands from mask
        for elemet in list_of_islands:
            connected_component_mask = connected_component_mask.remove_instance(elemet)

        # remove islands from graph
        for elem in np.unique(list_of_islands):
            self.neighborhood_graph_connected.remove_node(elem)

        get_logger().info("Removed number of islands: %s" % len(list_of_islands))
        get_logger().info("Number of RAG nodes: %s " % len(list(self.neighborhood_graph_connected.nodes)))

        return connected_component_mask, list_of_islands

    @staticmethod
    def get_rag(instance_segmentation: BioMedicalInstanceSegmentationMask, background: int = 0) -> RAG:
        """Gets the RegionAdjacencyGraph for an instance segmentation image.

        Args:
            instance_segmentation:
                The instance segmentation image.
            background:
                The background value.

        Returns:
            The RegionAdjacencyGraph.

        """
        rag = RAG(instance_segmentation.data)
        rag.remove_node(background)
        return rag

    @staticmethod
    def get_islands(neighborhood_graph: RAG) -> np.ndarray:
        list_of_islands = []
        for nodes in neighborhood_graph.nodes:
            if len(list(neighborhood_graph.neighbors(nodes))) == 0:
                list_of_islands.append(nodes)

        return np.unique(list_of_islands)


class SingleCellMasksCollection:
    """Collection of single cell masks and the corresponding label."""

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
