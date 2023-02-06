from __future__ import annotations

from hashlib import sha1
from typing import Dict, Union

import numpy as np

from polarityjam.model.parameter import ImageParameter
from polarityjam.model.masks import BioMedicalInstanceSegmentation, BioMedicalMask
from polarityjam.polarityjam_logging import get_logger


class BioMedicalChannel:  # todo: make it a PIL image for enhanced compatability?
    """Class representing a single channel of a biomedical image. It can contain multiple masks."""
    def __init__(self, channel: np.ndarray):
        self.data = channel
        self.masks: Dict[str, Union[BioMedicalMask, BioMedicalInstanceSegmentation]] = {}

    def mask(self, mask: BioMedicalMask) -> BioMedicalChannel:
        """Masks the channel with a given mask."""
        return BioMedicalChannel(self.data * mask.data)

    def add_mask(self, key: str, val: BioMedicalMask):
        """Adds a mask given a name to the channel."""
        self.masks[key] = val

    def remove_mask(self, key: str):
        """Removes a mask from the channel."""
        del self.masks[key]

    def get_mask_by_name(self, name: str) -> Union[BioMedicalMask, BioMedicalInstanceSegmentation]:
        """Returns a mask by its name."""
        return self.masks[name]


class BioMedicalImage:
    """Class representing a biomedical image. It can contain multiple channels and a segmentation."""
    def __init__(self, img: np.ndarray, img_params: ImageParameter,
                 segmentation: BioMedicalInstanceSegmentation = None):
        self.img = img
        self.segmentation = segmentation
        self.img_params = img_params
        self.marker = self.get_image_marker(img)
        self.junction = self.get_image_junction(img)
        self.nucleus = self.get_image_nucleus(img)
        self.organelle = self.get_image_organelle(img)
        self.img_hash = self.get_image_hash(img)

    def get_image_marker(self, img: np.ndarray):
        """Gets the image of the marker channel specified in the img_params.

        Args:
            img:
                The image to get the marker channel from.

        Returns:
            The np.ndarray of the marker channel.

        """
        if self.img_params.channel_expression_marker >= 0:
            get_logger().info("Marker channel at position: %s" % str(self.img_params.channel_expression_marker))
            return BioMedicalChannel(img[:, :, self.img_params.channel_expression_marker])
        return None

    def get_image_junction(self, img: np.ndarray):
        """Gets the image of the junction channel specified in the img_params.

        Args:
            img:
                The image to get the junction channel from.

        Returns:
            The np.ndarray of the junction channel.

        """
        if self.img_params.channel_junction >= 0:
            get_logger().info("Junction channel at position: %s" % str(self.img_params.channel_junction))
            return BioMedicalChannel(img[:, :, self.img_params.channel_junction])
        return None

    def get_image_nucleus(self, img: np.ndarray):
        """Gets the image of the nucleus channel specified in the img_params.

        Args:
            img:
                The image to get the nucleus channel from.

        Returns:
            The np.ndarray of the nucleus channel.

        """
        if self.img_params.channel_nucleus >= 0:
            get_logger().info("Nucleus channel at position: %s" % str(self.img_params.channel_nucleus))
            return BioMedicalChannel(img[:, :, self.img_params.channel_nucleus])
        return None

    def get_image_organelle(self, img: np.ndarray):
        """Gets the image of the organelle channel specified in the img_params.

        Args:
            img:
                The image to get the organelle channel from.

        Returns:
            The np.ndarray of the organelle channel.
        """
        if self.img_params.channel_organelle >= 0:
            get_logger().info("Organelle channel at position: %s" % str(self.img_params.channel_organelle))
            return BioMedicalChannel(img[:, :, self.img_params.channel_organelle])
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

    def has_nuclei(self) -> bool:
        """Returns whether the image has a nucleus channel."""
        return self.nucleus is not None

    def has_organelle(self) -> bool:
        """Returns whether the image has an organelle channel."""
        return self.organelle is not None

    def has_junction(self) -> bool:
        """Returns whether the image has a junction channel."""
        return self.junction is not None

    def has_marker(self) -> bool:
        """Returns whether the image has a marker channel."""
        return self.marker is not None
