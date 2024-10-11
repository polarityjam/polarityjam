"""Model classes for representing images and image data."""
from __future__ import annotations

from abc import ABC
from hashlib import sha1
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from polarityjam.compute.compute import contour_width
from polarityjam.compute.corner import get_contour
from polarityjam.compute.shape import center_single_cell, partition_single_cell_mask
from polarityjam.model.masks import BioMedicalInstanceSegmentation, BioMedicalMask
from polarityjam.model.parameter import ImageParameter, RuntimeParameter
from polarityjam.model.properties import (
    SingleCellJunctionProps,
    SingleCellMarkerCytosolProps,
    SingleCellMarkerMembraneProps,
    SingleCellMarkerNucleiProps,
    SingleCellMarkerProps,
    SingleCellNucleusProps,
    SingleCellOrganelleProps,
    SingleCellProps,
)
from polarityjam.polarityjam_logging import get_logger
from polarityjam.utils.normalization import normalize_arr


class BioMedicalChannel:  # todo: make it a PIL image for enhanced compatability?
    """Class representing a single channel of a biomedical image. It can contain multiple masks."""

    def __init__(
        self,
        channel: np.ndarray,
        masks: Optional[
            Dict[str, Union[BioMedicalMask, BioMedicalInstanceSegmentation]]
        ] = None,
    ):
        """Initialize the channel with the given data."""
        self.data = channel
        if masks is None:
            masks = {}
        self.masks = masks

    def mask(self, mask: BioMedicalMask) -> BioMedicalChannel:
        """Mask the channel with a given mask."""
        return BioMedicalChannel(self.data * mask.data)

    def add_mask(self, key: str, val: BioMedicalMask):
        """Add a mask given a name to the channel."""
        self.masks[key] = val

    def remove_mask(self, key: str):
        """Remove a mask from the channel."""
        del self.masks[key]

    def get_mask_by_name(
        self, name: str
    ) -> Union[BioMedicalMask, BioMedicalInstanceSegmentation]:
        """Return a mask by its name."""
        return self.masks[name]

    def normalize(
        self,
        source_limits: Optional[Tuple[Union[int, float], Union[int, float]]] = None,
        target_limits: Optional[Tuple[Union[int, float], Union[int, float]]] = None,
    ) -> BioMedicalChannel:
        """Normalize the channel data to a given range.

        Args:
            source_limits:
                The source limits to normalize from. E.g. (0, 255) for 8-bit (int) images.
            target_limits:
                The target limits to normalize to. E.g. (0, 1) for 64-bit (float) images.

        Returns:
            The normalized channel.

        """
        return BioMedicalChannel(
            normalize_arr(self.data, source_limits, target_limits), self.masks
        )


class AbstractBioMedicalImage(ABC):  # noqa: B024
    """Abstract class implementing basic functionality of a biomedical image."""

    @staticmethod
    def get_image_marker(
        img: np.ndarray, img_params: ImageParameter
    ) -> Optional[BioMedicalChannel]:
        """Get the image of the marker channel specified in the img_params.

        Args:
            img:
                The image to get the marker channel from.
            img_params:
                The image parameters.

        Returns:
            The np.ndarray of the marker channel.

        """
        if img_params.channel_expression_marker >= 0:
            get_logger().info(
                "Marker channel at position: %s"
                % str(img_params.channel_expression_marker)
            )
            return BioMedicalChannel(img[:, :, img_params.channel_expression_marker])
        return None

    @staticmethod
    def get_image_junction(
        img: np.ndarray, img_params: ImageParameter
    ) -> Optional[BioMedicalChannel]:
        """Get the image of the junction channel specified in the img_params.

        Args:
            img:
                The image to get the junction channel from.
            img_params:
                The image parameters.

        Returns:
            The np.ndarray of the junction channel.

        """
        if img_params.channel_junction >= 0:
            get_logger().info(
                "Junction channel at position: %s" % str(img_params.channel_junction)
            )
            return BioMedicalChannel(img[:, :, img_params.channel_junction])
        return None

    @staticmethod
    def get_image_nucleus(
        img: np.ndarray, img_params: ImageParameter
    ) -> Optional[BioMedicalChannel]:
        """Get the image of the nucleus channel specified in the img_params.

        Args:
            img:
                The image to get the nucleus channel from.
            img_params:
                The image parameters.

        Returns:
            The np.ndarray of the nucleus channel.

        """
        if img_params.channel_nucleus >= 0:
            get_logger().info(
                "Nucleus channel at position: %s" % str(img_params.channel_nucleus)
            )
            return BioMedicalChannel(img[:, :, img_params.channel_nucleus])
        return None

    @staticmethod
    def get_image_organelle(
        img: np.ndarray, img_params: ImageParameter
    ) -> Optional[BioMedicalChannel]:
        """Get the image of the organelle channel specified in the img_params.

        Args:
            img:
                The image to get the organelle channel from.
            img_params:
                The image parameters.

        Returns:
            The np.ndarray of the organelle channel.
        """
        if img_params.channel_organelle >= 0:
            get_logger().info(
                "Organelle channel at position: %s" % str(img_params.channel_organelle)
            )
            return BioMedicalChannel(img[:, :, img_params.channel_organelle])
        return None

    @staticmethod
    def get_image_hash(img: np.ndarray) -> str:
        """Return the hash of the given image.

        Args:
            img:
                The image to get the hash from.

        Returns:
            The hash of the image.

        """
        return sha1(img.copy(order="C")).hexdigest()  # type: ignore


class BioMedicalImage(AbstractBioMedicalImage):
    """Class representing a biomedical image.

    It can contain multiple channels and a segmentation.

    """

    def __init__(
        self,
        img: np.ndarray,
        img_params: ImageParameter,
        segmentation: Optional[BioMedicalInstanceSegmentation] = None,
    ):
        """Initialize the image with the given data."""
        self.img = img
        self.segmentation = segmentation
        self.img_params = img_params
        self.marker = self.get_image_marker(img, img_params)
        self.junction = self.get_image_junction(img, img_params)
        self.nucleus = self.get_image_nucleus(img, img_params)
        self.organelle = self.get_image_organelle(img, img_params)
        self.img_hash = self.get_image_hash(img)

    def has_nuclei(self) -> bool:
        """Return whether the image has a nucleus channel."""
        return self.nucleus is not None

    def has_organelle(self) -> bool:
        """Return whether the image has an organelle channel."""
        return self.organelle is not None

    def has_junction(self) -> bool:
        """Return whether the image has a junction channel."""
        return self.junction is not None

    def has_marker(self) -> bool:
        """Return whether the image has a marker channel."""
        return self.marker is not None

    def focus(
        self,
        connected_component_label: int,
        membrane_thickness: int,
        junction_threshold: Optional[float] = None,
    ) -> SingleCellImage:
        """Focus the image on a given connected component label.

        Args:
            connected_component_label:
                The connected component label to focus on.
            membrane_thickness:
                The thickness of the membrane to add around the cell.

        Returns:
            The focused image.
        """
        (
            sc_mask,
            sc_membrane_mask,
            sc_nucleus_mask,
            sc_cytosol_mask,
            sc_organelle_mask,
            sc_junction_protein_mask,
        ) = self.get_single_cell_masks(
            connected_component_label, membrane_thickness, junction_threshold
        )

        sc_mask_f = np.flip(sc_mask.data, axis=0)  # y-axis is flipped
        contour = get_contour(sc_mask_f.astype(int))

        return SingleCellImage(
            self,
            contour=contour,
            connected_component_label=connected_component_label,
            single_cell_mask=sc_mask,
            single_cell_membrane_mask=sc_membrane_mask,
            single_nucleus_mask=sc_nucleus_mask,
            single_organelle_mask=sc_organelle_mask,
            single_junction_mask=sc_junction_protein_mask,
            single_cytosol_mask=sc_cytosol_mask,
        )

    def get_single_cell_masks(
        self,
        connected_component_label: int,
        membrane_thickness: int,
        junction_threshold: Optional[float] = None,
    ) -> Tuple[
        BioMedicalMask,
        BioMedicalMask,
        Optional[BioMedicalMask],
        Optional[BioMedicalMask],
        Optional[BioMedicalMask],
        Optional[BioMedicalMask],
    ]:
        """Return all masks of the image.

        Args:
            connected_component_label:
                The connected component label to focus on.
            membrane_thickness:
                The thickness of the membrane to add around the cell.

        Returns:
            The masks of the image.

        """
        assert self.segmentation is not None, "The image has no segmentation."

        sc_mask = self.get_single_cell_mask(connected_component_label)
        sc_membrane_mask = self.get_single_membrane_mask(
            connected_component_label, membrane_thickness
        )

        (
            sc_nucleus_mask,
            sc_cytosol_mask,
            sc_organelle_mask,
            sc_junction_protein_mask,
        ) = (None, None, None, None)
        if self.has_nuclei():
            sc_nucleus_mask = self.get_single_nuclei_mask(connected_component_label)
            sc_cytosol_mask = self.get_single_cytosol_mask(connected_component_label)

        if self.has_organelle():
            sc_organelle_mask = self.get_single_organelle_mask(
                connected_component_label
            )

        if self.has_junction():
            sc_junction_protein_mask = self.get_single_junction_mask(
                connected_component_label,
                membrane_thickness,
                junction_threshold,
            )

        return (
            sc_mask,
            sc_membrane_mask,
            sc_nucleus_mask,
            sc_cytosol_mask,
            sc_organelle_mask,
            sc_junction_protein_mask,
        )

    def get_single_cell_mask(self, connected_component_label: int) -> BioMedicalMask:
        """Get the mask of the single cell.

        Args:
            connected_component_label:
                The connected component label to focus on.

        Returns:
            The mask of the single cell.

        """
        assert self.segmentation is not None, "The image has no segmentation."
        sc_mask = (
            self.segmentation.segmentation_mask_connected.get_single_instance_mask(
                connected_component_label
            )
        )
        return sc_mask

    def get_single_membrane_mask(
        self, connected_component_label: int, membrane_thickness: int
    ) -> BioMedicalMask:
        """Get the mask of the single cell membrane.

        Args:
            connected_component_label:
                The connected component label to focus on.
            membrane_thickness:
                The thickness of the membrane.

        Returns:
            The mask of the single cell membrane.

        """
        assert self.segmentation is not None, "The image has no segmentation."

        sc_mask = self.get_single_cell_mask(connected_component_label)
        sc_membrane_mask = sc_mask.get_outline_from_mask(membrane_thickness)

        if self.segmentation.segmentation_mask_junction is not None:
            assert self.segmentation.segmentation_mask_junction is not None

            # get membrane mask from junction mask
            sc_membrane_mask = (
                self.segmentation.segmentation_mask_junction.get_single_instance_mask(
                    connected_component_label
                ).operation(sc_membrane_mask, np.logical_or)
            )

        return sc_membrane_mask

    def get_single_nuclei_mask(self, connected_component_label: int) -> BioMedicalMask:
        """Get the mask of the single cell nucleus.

        Args:
            connected_component_label:
                The connected component label to focus on.

        Returns:
            The mask of the single cell nucleus.

        """
        assert self.segmentation is not None, "The image has no segmentation."
        assert (
            self.segmentation.segmentation_mask_nuclei is not None
        ), "The image has no nuclei segmentation."
        sc_nucleus_mask = (
            self.segmentation.segmentation_mask_nuclei.get_single_instance_mask(
                connected_component_label
            )
        )
        return sc_nucleus_mask

    def get_single_cytosol_mask(
        self, connected_component_label: int
    ) -> Optional[BioMedicalMask]:
        """Get the mask of the single cell cytosol.

        Args:
            connected_component_label:
                The connected component label to focus on.

        Returns:
            The mask of the single cell cytosol.

        """
        assert self.segmentation is not None, "The image has no segmentation."
        assert (
            self.segmentation.segmentation_mask_nuclei is not None
        ), "The image has no nuclei segmentation."
        sc_mask = self.get_single_cell_mask(connected_component_label)
        sc_nucleus_mask = self.get_single_nuclei_mask(connected_component_label)
        sc_cytosol_mask = sc_nucleus_mask.operation(sc_mask, np.logical_xor)

        return sc_cytosol_mask

    def get_single_organelle_mask(
        self, connected_component_label: int
    ) -> Optional[BioMedicalMask]:
        """Get the mask of the single cell organelle.

        Args:
            connected_component_label:
                The connected component label to focus on.

        Returns:
            The mask of the single cell organelle.

        """
        assert self.segmentation is not None, "The image has no segmentation."
        assert (
            self.segmentation.segmentation_mask_organelle is not None
        ), "The image has no organelle segmentation."
        sc_organelle_mask = (
            self.segmentation.segmentation_mask_organelle.get_single_instance_mask(
                connected_component_label
            )
        )
        return sc_organelle_mask

    def get_single_junction_mask(
        self,
        connected_component_label: int,
        membrane_thickness: int,
        junction_threshold: Optional[float] = None,
    ) -> BioMedicalMask:
        """Get the mask of the single cell junction protein.

        Args:
            connected_component_label:
                The connected component label to focus on.
            membrane_thickness:
                The thickness of the membrane.

        Returns:
            The mask of the single cell junction protein.

        """
        assert self.segmentation is not None, "The image has no segmentation."
        assert self.junction is not None, "The image has no junction channel."

        # segmentation not necessarily given and cannot be computed on whole image
        if self.segmentation.segmentation_mask_junction is None:
            sc_junction_protein_mask = (
                self._calc_default_single_cell_junction_segmentation(
                    connected_component_label,
                    membrane_thickness,
                    junction_threshold,
                )
            )
        else:  # case mask given as BioMedicalJunctionSegmentation
            # todo: future concept of BioMedicalJunctionSegmentation
            sc_junction_protein_mask = (
                self.segmentation.segmentation_mask_junction.get_single_instance_mask(
                    connected_component_label
                )
            )

        return sc_junction_protein_mask

    def _calc_default_single_cell_junction_segmentation(
        self,
        connected_component_label,
        membrane_thickness,
        junction_threshold,
    ):
        """Calculate single cell junction protein mask."""
        assert self.junction is not None, "The image has no junction channel."

        # we cannot compute the segmentation on the whole junction channel,
        # hence we calculate it now on the single cell membrane mask
        sc_membrane_mask = self.get_single_membrane_mask(
            connected_component_label, membrane_thickness
        )
        masked_sc_junction_channel = self.junction.mask(sc_membrane_mask)

        if junction_threshold is not None and junction_threshold > 0:
            # manual threshold
            sc_junction_protein_mask = BioMedicalMask.from_threshold(
                masked_sc_junction_channel.data, junction_threshold
            )
        else:
            # auto threshold
            sc_junction_protein_mask = BioMedicalMask.from_threshold_otsu(
                masked_sc_junction_channel.data, gaussian_filter=None
            )
        return sc_junction_protein_mask


class SingleCellImage(AbstractBioMedicalImage):
    """Class representing a single cell image."""

    def __init__(
        self,
        img: BioMedicalImage,
        contour: np.ndarray,
        connected_component_label: int,
        single_cell_mask: BioMedicalMask,
        single_cell_membrane_mask: BioMedicalMask,
        single_nucleus_mask: Optional[BioMedicalMask] = None,
        single_organelle_mask: Optional[BioMedicalMask] = None,
        single_junction_mask: Optional[BioMedicalMask] = None,
        single_cytosol_mask: Optional[BioMedicalMask] = None,
    ):
        """Initialize the single cell image."""
        self.contour = contour
        self.img = img
        self.contour_width = self.get_contour_width()
        self.connected_component_label = connected_component_label

        self.cell_mask = single_cell_mask
        self.cell_membrane_mask = single_cell_membrane_mask
        self.nucleus_mask = single_nucleus_mask
        self.organelle_mask = single_organelle_mask
        self.junction_mask = single_junction_mask
        self.cytosol_mask = single_cytosol_mask

    def get_cell_properties(self, param: RuntimeParameter) -> SingleCellProps:
        """Get the properties of the single cell.

        Args:
            param:
                The runtime parameter.

        Returns:
            The properties of the single cell.

        """
        return SingleCellProps(
            self.cell_mask,
            self.center_mask(self.cell_mask),
            param.dp_epsilon,
            param.cue_direction,
        )

    def get_nucleus_properties(self, param: RuntimeParameter) -> SingleCellNucleusProps:
        """Get the properties of the single cell nucleus.

        Args:
            param:
                The runtime parameter.

        Returns:
            The properties of the single cell nucleus.

        """
        assert self.nucleus_mask is not None, "No nucleus mask provided."
        return SingleCellNucleusProps(
            self.nucleus_mask, self.get_cell_properties(param)
        )

    def get_organelle_properties(
        self, param: RuntimeParameter
    ) -> SingleCellOrganelleProps:
        """Get the properties of the single cell organelle.

        Args:
            param:
                The runtime parameter.

        Returns:
            The properties of the single cell organelle.

        """
        assert self.organelle_mask is not None, "No organelle mask provided."
        return SingleCellOrganelleProps(
            self.organelle_mask, self.get_nucleus_properties(param)
        )

    def get_marker_properties(self, param: RuntimeParameter) -> SingleCellMarkerProps:
        """Get the properties of the single cell marker.

        Args:
            param:
                The runtime parameter.

        Returns:
            The properties of the single cell marker.

        """
        assert self.img.marker is not None, "The image has no marker channel."
        return SingleCellMarkerProps(
            self.cell_mask,
            self.img.marker,
            half_cell_masks=self.half_mask(param.cue_direction),
            quadrant_cell_masks=self.quarter_mask(param.cue_direction),
        )

    def get_marker_membrane_properties(
        self, param: RuntimeParameter
    ) -> SingleCellMarkerMembraneProps:
        """Get the properties of the single cell marker membrane.

        Args:
            param:
                The runtime parameter.

        Returns:
            The properties of the single cell marker membrane.

        """
        assert self.img.marker is not None, "The image has no marker channel."
        return SingleCellMarkerMembraneProps(self.cell_membrane_mask, self.img.marker)

    def get_marker_nucleus_properties(
        self, param: RuntimeParameter
    ) -> SingleCellMarkerNucleiProps:
        """Get the properties of the single cell marker nucleus.

        Args:
            param:
                The runtime parameter.

        Returns:
            The properties of the single cell marker nucleus.

        """
        assert self.nucleus_mask is not None, "No nucleus mask provided."
        assert self.img.marker is not None, "The image has no marker channel."

        return SingleCellMarkerNucleiProps(
            self.nucleus_mask,
            self.img.marker,
            self.get_nucleus_properties(param),
            self.get_marker_properties(param),
        )

    def get_marker_cytosol_properties(
        self, param: RuntimeParameter
    ) -> SingleCellMarkerCytosolProps:
        """Get the properties of the single cell marker cytosol.

        Args:
            param:
                The runtime parameter.

        Returns:
            The properties of the single cell marker cytosol.

        """
        assert self.img.marker is not None, "The image has no marker channel."
        assert self.cytosol_mask is not None, "No cytosol mask provided."

        return SingleCellMarkerCytosolProps(
            self.cytosol_mask,
            self.img.marker,
            self.get_marker_nucleus_properties(param),
        )

    def get_junction_properties(
        self, param: RuntimeParameter
    ) -> SingleCellJunctionProps:
        """Get the properties of the single cell junction.

        Args:
            param:
                The runtime parameter.

        Returns:
            The properties of the single cell junction.

        """
        assert self.img.junction is not None, "The image has no junction channel."
        assert self.junction_mask is not None, "No junction mask provided."

        return SingleCellJunctionProps(
            im_junction=self.img.junction,
            single_cell_membrane_mask=self.cell_membrane_mask,
            single_cell_junction_intensity_mask=self.junction_mask,
            single_cell_mask=self.cell_mask,
            single_cell_props=self.get_cell_properties(param),
            half_masks=self.half_mask(param.cue_direction),
            quadrant_masks=self.quarter_mask(param.cue_direction),
            cue_direction=param.cue_direction,
            dp_epsilon=param.dp_epsilon,
        )

    def center_mask(self, mask: BioMedicalMask) -> BioMedicalMask:
        """Center the mask on the contour.

        Args:
            mask:
                The mask to be centered.

        Returns:
            The centered mask.

        """
        return BioMedicalMask(center_single_cell([mask], self.contour)[0])

    def center_channel(self, channel: BioMedicalChannel) -> BioMedicalChannel:
        """Center the channel on the contour.

        Args:
            channel:
                The channel to be centered.

        Returns:
            The centered channel.

        """
        return BioMedicalChannel(center_single_cell([channel], self.contour)[0])

    def get_contour_width(self) -> float:
        """Get the width of the contour."""
        return contour_width(self.contour)

    def half_mask(self, cue_direction) -> List[BioMedicalMask]:
        """Get the half masks of the single cell.

        Args:
            cue_direction:
                The cue direction.

        Returns:
            The half mask of the single cell.

        """
        half_masks_, _, _ = partition_single_cell_mask(
            self.cell_mask.operation(self.cell_membrane_mask, np.logical_or),
            cue_direction,
            self.contour_width,
            2,
            self.contour,
        )
        half_masks = [BioMedicalMask(half_mask_) for half_mask_ in half_masks_]
        return half_masks

    def quarter_mask(self, cue_direction) -> List[BioMedicalMask]:
        """Get the quarter masks of the single cell.

        Args:
            cue_direction:
                The cue direction.

        Returns:
            The quarters of the single cell.

        """
        quadrant_masks_, _, _ = partition_single_cell_mask(
            self.cell_mask.operation(self.cell_membrane_mask, np.logical_or),
            cue_direction,
            self.contour_width,
            4,
            self.contour,
        )
        quadrant_mask = [
            BioMedicalMask(quadrant_mask_) for quadrant_mask_ in quadrant_masks_
        ]
        return quadrant_mask

    def has_nuclei(self) -> bool:
        """Return whether the image has a nucleus channel."""
        return self.img.nucleus is not None

    def has_organelle(self) -> bool:
        """Return whether the image has an organelle channel."""
        return self.img.organelle is not None

    def has_junction(self) -> bool:
        """Return whether the image has a junction channel."""
        return self.img.junction is not None

    def has_marker(self) -> bool:
        """Return whether the image has a marker channel."""
        return self.img.marker is not None

    def threshold_cell_size(self, threshold: int) -> bool:
        """Return whether the cell size is above the threshold.

        Args:
            threshold: The threshold for the cell size.

        """
        if self.cell_mask.is_count_below_threshold(threshold):
            return True
        return False

    def threshold_organelle_size(self, threshold: int) -> bool:
        """Return whether the organelle size is above the threshold.

        Args:
            threshold:
                The threshold for the organelle size.

        Returns:
            True if the organelle size is above the threshold, False otherwise.

        """
        if (
            self.organelle_mask is not None
            and self.organelle_mask.is_count_below_threshold(threshold)
        ):
            return True
        return False

    def threshold_nucleus_size(self, threshold: int) -> bool:
        """Return whether the nucleus size is above the threshold.

        Args:
            threshold:
                The threshold for the nucleus size.

        Returns:
            True if the nucleus size is above the threshold, False otherwise.

        """
        if (
            self.nucleus_mask is not None
            and self.nucleus_mask.is_count_below_threshold(threshold)
        ):
            return True
        return False
