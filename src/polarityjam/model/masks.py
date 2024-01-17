"""Classes to represent masks for the image data."""
from __future__ import annotations

import copy
import warnings
from typing import Any, Callable, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.future.graph import RAG
from skimage.morphology import remove_small_objects as rmso
from skimage.restoration import rolling_ball
from skimage.segmentation import clear_border as clear_border_skimage

_T = TypeVar("_T")


class Mask:
    """Class representing a base mask."""

    def __init__(self, mask: np.ndarray):
        """Initialize the mask with the given data."""
        self._warn = False
        self.data = mask.astype(bool)

        if self._warn and self.data.sum() == 0:
            warnings.warn("Mask is empty!")

    def warn(self, warn: bool) -> None:
        """Set the warning flag.

        Args:
            warn:
                The warning flag.
        """
        self._warn = warn

    @classmethod
    def from_threshold_otsu(
        cls: Type[_T],
        channel: np.ndarray,
        gaussian_filter=None,
        rolling_ball_radius=None,
        median_filter=None,
    ) -> _T:
        """Initialize a mask from a channel using Otsu's method."""
        if gaussian_filter is not None:
            img_channel_blur = ndi.gaussian_filter(channel, sigma=3)
        else:
            img_channel_blur = channel

        if rolling_ball_radius is not None:
            img_channel_blur = rolling_ball(
                img_channel_blur, radius=rolling_ball_radius
            )

        if median_filter is not None:
            img_channel_blur = ndi.median_filter(img_channel_blur, size=median_filter)

        interior_mask = np.where(
            img_channel_blur > threshold_otsu(img_channel_blur),
            True,
            False,
        )

        return cls(interior_mask)  # type: ignore [call-arg]

    @classmethod
    def from_threshold(cls: Type[_T], channel: np.ndarray, threshold: float) -> _T:
        """Initialize a mask from a channel using a manual threshold."""
        interior_mask = np.where(
            channel > threshold,
            True,
            False,
        )

        return cls(interior_mask)  # type: ignore [call-arg]

    @classmethod
    def empty(cls: Type[_T], shape: Tuple[int, ...]) -> _T:
        """Create an empty mask of a given shape."""
        return cls(np.zeros(shape))  # type: ignore [call-arg]

    def is_count_below_threshold(self, pixel_value: int) -> bool:
        """Check if the number of pixels in the mask is below a given threshold.

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
        """Initialize a mask from a numpy array."""
        super().__init__(mask)

    def invert(self) -> BioMedicalMask:
        """Invert the mask."""
        return BioMedicalMask(np.invert(self.data))

    def operation(self, overlay: BioMedicalMask, operator: Callable) -> BioMedicalMask:
        """Perform an operation on the mask with another mask."""
        return BioMedicalMask(
            operator(overlay.data.astype(bool), self.data.astype(bool))
        )

    def get_outline_from_mask(self, width: int = 1) -> BioMedicalMask:
        """Compute outline for a single cell mask.

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
        self, connected_component_mask: BioMedicalInstanceSegmentationMask
    ) -> BioMedicalInstanceSegmentationMask:
        """Overlay an instance segmentation mask on the mask."""
        return BioMedicalInstanceSegmentationMask(
            self.data * connected_component_mask.data
        )

    def to_instance_mask(
        self, instance_label: int = 1
    ) -> BioMedicalInstanceSegmentationMask:
        """Convert the mask to an instance segmentation mask."""
        return BioMedicalInstanceSegmentationMask(self.data * instance_label)

    def mask_background(
        self,
    ) -> BioMedicalMask:
        """Mask the background of the mask."""
        return BioMedicalMask(
            np.ma.masked_where(self.data == False, self.data)  # noqa: E712
        )

    def combine(self, other_mask: BioMedicalMask) -> BioMedicalMask:
        """Combine (AND) the mask with another mask."""
        return BioMedicalMask(np.logical_and(self.data, other_mask.data))

    def disjoin(self, other_mask: BioMedicalMask) -> BioMedicalMask:
        """Logical disjunction (XOR) from the other mask and this mask."""
        return BioMedicalMask(np.logical_xor(self.data, other_mask.data))


class BioMedicalInstanceSegmentationMask(Mask):
    """Class representing an instance segmentation mask."""

    def __init__(
        self,
        mask: np.ndarray,
        dtype: Union[str, Any] = int,
        background_label: Union[int, float] = 0,
    ):
        """Initialize a mask from a numpy array."""
        super().__init__(mask)
        self.background_label = background_label
        self.data = mask.astype(dtype)

    def remove_instance(
        self, instance_label: int
    ) -> BioMedicalInstanceSegmentationMask:
        """Remove a single instance from the mask."""
        return BioMedicalInstanceSegmentationMask(
            np.where(self.data == instance_label, self.background_label, self.data),
            background_label=self.background_label,
        )

    def clear_border(self) -> BioMedicalInstanceSegmentationMask:
        """Clear the border of the mask."""
        return BioMedicalInstanceSegmentationMask(clear_border_skimage(self.data))

    def remove_small_objects(
        self, min_size: int, connectivity=2
    ) -> BioMedicalInstanceSegmentationMask:
        """Remove small objects from the mask."""
        return BioMedicalInstanceSegmentationMask(
            rmso(self.data, min_size, connectivity=connectivity)
        )

    def mask_background(
        self,
        background_label: Union[int, float] = 0,
        mask_with: Optional[
            Union[np.ndarray, BioMedicalInstanceSegmentationMask]
        ] = None,
    ) -> BioMedicalInstanceSegmentationMask:
        """Mask the background of the mask."""
        if mask_with is None:
            mask_with = self.data
        elif isinstance(mask_with, BioMedicalInstanceSegmentationMask):
            mask_with = mask_with.data

        return BioMedicalInstanceSegmentationMask(
            np.ma.masked_where(self.data == background_label, mask_with),
            dtype=float,
            background_label=self.background_label,
        )

    def get_single_instance_mask(self, instance_label: int) -> BioMedicalMask:
        """Get the single cell mask given its label.

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
        """Return the number of instances in the mask."""
        return len(np.unique(self.data))

    def get_labels(self, exclude_background: bool = True) -> List[int]:
        """Get all labels of the mask.

        Args:
            exclude_background:
                Whether to exclude the background label.

        Returns:
            The labels of the mask.

        """
        labels = np.unique(self.data)

        if exclude_background:
            labels = labels[labels != self.background_label]

        return list(labels)

    def to_semantic_mask(self) -> BioMedicalMask:
        """Convert the mask to a semantic (boolean) mask."""
        return BioMedicalMask(self.data.astype(bool))

    def element_mult(
        self, mask: Union[BioMedicalInstanceSegmentationMask, np.ndarray]
    ) -> BioMedicalInstanceSegmentationMask:
        """Perform an element-wise multiplication of the mask with another mask.

        Args:
            mask:
                The mask to multiply with.

        Returns:
            The resulting mask.

        """
        if isinstance(mask, BioMedicalInstanceSegmentationMask):
            mask = mask.data
        return BioMedicalInstanceSegmentationMask(
            self.data * mask, dtype=float, background_label=self.background_label
        )

    def element_add(
        self, mask: Union[BioMedicalInstanceSegmentationMask, np.ndarray]
    ) -> BioMedicalInstanceSegmentationMask:
        """Perform an element-wise addition of the mask with another mask.

        Args:
            mask:
                The mask to add with.

        Returns:
            The resulting mask.

        """
        if isinstance(mask, BioMedicalInstanceSegmentationMask):
            mask = mask.data
        return BioMedicalInstanceSegmentationMask(
            self.data + mask, dtype=float, background_label=self.background_label
        )

    def scalar_mult(self, scalar: int) -> BioMedicalInstanceSegmentationMask:
        """Perform a scalar multiplication of the mask.

        Args:
            scalar:
                The scalar to multiply with.

        Returns:
            The resulting mask.

        """
        return BioMedicalInstanceSegmentationMask(
            self.data * scalar, dtype=float, background_label=self.background_label
        )

    def scalar_add(self, scalar: int) -> BioMedicalInstanceSegmentationMask:
        """Perform a scalar addition of the mask.

        Args:
            scalar:
                The scalar to add with.

        Returns:
            The resulting mask.

        """
        return BioMedicalInstanceSegmentationMask(
            self.data + scalar, dtype=float, background_label=self.background_label
        )

    def relabel(
        self, new_labels: Union[dict, np.ndarray], exclude_background: bool = True
    ) -> BioMedicalInstanceSegmentationMask:
        """Reallocate the labels of the mask.

        Args:
            new_labels:
                A dictionary mapping old labels to new labels.
                Alternatively, a numpy array that assumes consecutive ordered labels.
            exclude_background:
                Whether to exclude the background label from the reallocation.

        Returns:
            New mask object with a reallocated mask.

        """
        num_instances = len(self) - 1 if exclude_background else len(self)

        old_labels = np.unique(self.data)

        if exclude_background:
            old_labels = old_labels[old_labels != self.background_label]

        if isinstance(new_labels, dict):
            new_labels = np.array([new_labels[label] for label in old_labels])

        err_str = """The number of new labels must match the number of old labels.
            Got new label length {len_a} and old label length {len_b}.""".format(
            len_a=len(new_labels), len_b=num_instances
        )
        assert len(new_labels) == num_instances, err_str

        new_mask = np.zeros_like(self.data).astype(new_labels.dtype)
        for old_label, new_label in zip(old_labels, new_labels):
            new_mask[self.data == old_label] = new_label

        return BioMedicalInstanceSegmentationMask(
            new_mask, dtype=new_mask.dtype, background_label=self.background_label
        )


class BioMedicalJunctionSegmentation:
    """Class for a junction segmentation of an image."""

    def __init__(
        self,
        junction_masks,
        junction_ids,
        junction_cell_ids,
    ):
        """Initialize a junction segmentation from a mask."""
        self.junction_masks = junction_masks
        self.junction_ids = junction_ids
        self.junction_cell_ids = junction_cell_ids

    def get_single_instance_mask(self, instance_label) -> BioMedicalMask:
        """Get the single cell mask given its label."""
        return BioMedicalMask(
            self.get_junction_by_cell_ids(self.get_junction_cell_ids(instance_label))
        )

    def get_junction_cell_ids(self, cell_id):
        """Get the junction ids of a cell."""
        edges = np.sum(self.junction_cell_ids == cell_id, axis=0)
        return self.junction_ids[edges.astype(bool)]

    def get_junction_by_cell_ids(self, junction_cell_ids):
        """Get the junction mask of a cell."""
        lab = 0 * np.copy(self.junction_masks)

        for inst_id in junction_cell_ids:
            inst = (self.junction_masks == inst_id).astype(bool)
            lab[inst] = 1
        return lab

    def remove_instance(self, instance_label: int) -> BioMedicalJunctionSegmentation:
        """Remove an instance from the mask.

        Args:
            instance_label:
                The instance label to remove.

        Returns:
            New mask object with the instance removed.

        """
        new_junction_masks = np.copy(self.junction_masks)
        new_junction_ids = np.copy(self.junction_ids)
        new_junction_cell_ids = np.copy(self.junction_cell_ids)

        instance_id_mask_cell_1 = self.junction_cell_ids[0, :] == instance_label
        instance_id_mask_cell_2 = self.junction_cell_ids[1, :] == instance_label

        # set cell_ids to 0
        new_junction_cell_ids[0, instance_id_mask_cell_1] = 0
        new_junction_cell_ids[1, instance_id_mask_cell_2] = 0

        # remove junction id iff both cell_ids are 0
        x = []
        [
            x.append(idx)  # type: ignore
            for idx in range(self.junction_cell_ids.shape[1])
            if self.junction_cell_ids[:, idx].sum() == 0
        ]

        for en_idx, idx in enumerate(x):
            # index offset
            idx = idx - en_idx

            # get boolean mask
            ind = np.ones(new_junction_cell_ids.shape[1], bool)
            ind[idx] = False
            new_junction_cell_ids = new_junction_cell_ids[:, ind]

            # delete based on boolean mask
            junction_id_to_delete = new_junction_ids[idx]
            new_junction_ids = new_junction_ids[ind]

            # remove junction mask values
            new_junction_masks[new_junction_masks == junction_id_to_delete] = 0

        return BioMedicalJunctionSegmentation(
            new_junction_masks, new_junction_ids, new_junction_cell_ids
        )


class BioMedicalInstanceSegmentation:
    """Class holding entire biomedical image instance segmentations of various components."""

    def __init__(
        self,
        segmentation_mask: BioMedicalInstanceSegmentationMask,
        connection_graph: bool = True,
        segmentation_mask_nuclei: Optional[BioMedicalInstanceSegmentationMask] = None,
        segmentation_mask_organelle: Optional[
            BioMedicalInstanceSegmentationMask
        ] = None,
        segmentation_mask_junction: Optional[BioMedicalJunctionSegmentation] = None,
    ):
        """Initialize an instance segmentation from a mask."""
        self.segmentation_mask = segmentation_mask
        self._segmentation_mask_nuclei = None
        self._segmentation_mask_organelle = None
        self._segmentation_mask_junction = None

        self.connection_graph = connection_graph

        self.neighborhood_graph = BioMedicalInstanceSegmentation.get_rag(
            self.segmentation_mask
        )

        self.island_list: List[int] = []
        self.segmentation_mask_connected: BioMedicalInstanceSegmentationMask = (
            copy.deepcopy(self.segmentation_mask)
        )
        self.neighborhood_graph_connected = copy.deepcopy(self.neighborhood_graph)

        if segmentation_mask_nuclei is not None:
            # assure same labels
            self.segmentation_mask_nuclei = segmentation_mask_nuclei

        if segmentation_mask_organelle is not None:
            # assure same labels
            self.segmentation_mask_organelle = segmentation_mask_organelle

        if segmentation_mask_junction is not None:
            self.segmentation_mask_junction = segmentation_mask_junction

        if self.connection_graph:
            self.update_graphs()

    @property
    def segmentation_mask_nuclei(self):
        """Get the nuclei segmentation mask."""
        return self._segmentation_mask_nuclei

    @segmentation_mask_nuclei.setter
    def segmentation_mask_nuclei(self, value: BioMedicalInstanceSegmentationMask):
        """Set the nuclei segmentation mask."""
        if value is not None:
            self._segmentation_mask_nuclei = (
                value.to_semantic_mask().overlay_instance_segmentation(
                    self.segmentation_mask_connected
                )
            )

    @property
    def segmentation_mask_organelle(self):
        """Get the organelle segmentation mask."""
        return self._segmentation_mask_organelle

    @segmentation_mask_organelle.setter
    def segmentation_mask_organelle(self, value: BioMedicalInstanceSegmentationMask):
        """Set the organelle segmentation mask."""
        if value is not None:
            self._segmentation_mask_organelle = (
                value.to_semantic_mask().overlay_instance_segmentation(
                    self.segmentation_mask_connected
                )
            )

    @property
    def segmentation_mask_junction(self):
        """Get the junction segmentation mask."""
        return self._segmentation_mask_junction

    @segmentation_mask_junction.setter
    def segmentation_mask_junction(self, value: BioMedicalJunctionSegmentation):
        """Set the junction segmentation mask."""
        if value is not None:
            # todo: how to assure same labels?
            self._segmentation_mask_junction = value

    def update_graphs(self) -> List[int]:
        """Update the graphs after a change in the segmentation mask.

        Returns:
            island_list

        """
        if not self.connection_graph:
            warnings.warn("Connection graph is disabled.")
            return []

        island_list = self.remove_islands()

        self.island_list.extend(island_list)

        return island_list

    def remove_instance_label(self, instance_label) -> List[int]:
        """Remove an instance label from the segmentation inplace.

        Args:
            instance_label:
                The instance label to remove.

        Returns:
            All cells that have been additionally removed because they were not connected to the main graph anymore.

        """
        self.neighborhood_graph.remove_node(instance_label)
        self.neighborhood_graph_connected.remove_node(instance_label)
        self.segmentation_mask = self.segmentation_mask.remove_instance(instance_label)
        self.segmentation_mask_connected = (
            self.segmentation_mask_connected.remove_instance(instance_label)
        )

        # keep optional masks in synchronization
        if self.segmentation_mask_nuclei is not None:
            self.segmentation_mask_nuclei = (
                self.segmentation_mask_nuclei.remove_instance(instance_label)
            )
        if self.segmentation_mask_organelle is not None:
            self.segmentation_mask_organelle = (
                self.segmentation_mask_organelle.remove_instance(instance_label)
            )

        if self.segmentation_mask_junction is not None:
            self.segmentation_mask_junction = (
                self.segmentation_mask_junction.remove_instance(instance_label)
            )

        # remove all cells that are not connected to the main graph anymore
        if self.connection_graph:
            return self.update_graphs()

        return []

    def set_feature_of_interest(
        self, feature_of_interest_name: str, feature_of_interest_vec: np.ndarray
    ):
        """Set the feature of interest for the segmentation.

        Args:
            feature_of_interest_name:
                The name of the feature of interest.
            feature_of_interest_vec:
                The feature of interest vector.

        """
        nodes = sorted(self.neighborhood_graph_connected.nodes)

        assert len(feature_of_interest_vec) == len(
            nodes
        ), "The length of the feature of interest vector must match the number of nodes in the graph."

        for idx, node in enumerate(nodes):
            self.neighborhood_graph_connected.nodes[node][
                feature_of_interest_name
            ] = feature_of_interest_vec[idx]

    def remove_islands(self) -> List[int]:
        """Remove unconnected cells from the mask (Cells without neighbours).

        Returns:
            A list of unconnected cells (Islands) that have been removed.

        """
        if not self.connection_graph:
            warnings.warn("Connection graph is disabled.")
            return []

        # Get list of islands - nodes with no neighbours
        list_of_islands = BioMedicalInstanceSegmentation.get_islands(
            self.neighborhood_graph_connected
        )

        # remove islands from masks
        for element in np.unique(list_of_islands):
            # remove islands from graph
            self.neighborhood_graph_connected.remove_node(element)

            # connected segmentation mask
            self.segmentation_mask_connected = (
                self.segmentation_mask_connected.remove_instance(element)
            )

            # nuclei mask
            if self.segmentation_mask_nuclei is not None:
                self.segmentation_mask_nuclei = (
                    self.segmentation_mask_nuclei.remove_instance(element)
                )

            # organelle mask
            if self.segmentation_mask_organelle is not None:
                self.segmentation_mask_organelle = (
                    self.segmentation_mask_organelle.remove_instance(element)
                )

            # junction mask
            if self.segmentation_mask_junction is not None:
                self.segmentation_mask_junction = (
                    self.segmentation_mask_junction.remove_instance(element)
                )

        return list_of_islands

    @staticmethod
    def get_rag(
        instance_segmentation: BioMedicalInstanceSegmentationMask, background: int = 0
    ) -> RAG:
        """Get the RegionAdjacencyGraph for an instance segmentation image.

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
    def get_islands(neighborhood_graph: RAG) -> List[int]:
        """Get a list of islands (nodes with no neighbours) from a neighborhood graph.

        Args:
            neighborhood_graph:
                The neighborhood graph.

        Returns:
            A list of islands.

        """
        list_of_islands = []
        for nodes in neighborhood_graph.nodes:
            if len(list(neighborhood_graph.neighbors(nodes))) == 0:
                list_of_islands.append(nodes)

        return list(np.unique(list_of_islands))
