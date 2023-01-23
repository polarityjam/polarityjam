from hashlib import sha1

import numpy as np
import skimage

from polarityjam import ImageParameter
from polarityjam.polarityjam_logging import get_logger
from polarityjam.model.masks import BioMedicalInstanceSegmentation, BioMedicalMask


class BioMedicalChannel:  # todo: make it a PIL image for enhanced compatability?
    def __init__(self, channel: np.ndarray):
        self.channel = channel

    def threshold_otsu(self):
        otsu_val = skimage.filters.threshold_otsu(self.channel)
        channel = np.copy(self.channel)
        channel[self.channel <= otsu_val] = 0
        channel[self.channel > otsu_val] = 1
        return BioMedicalMask(channel)

    def mask(self, mask: BioMedicalMask):
        return BioMedicalChannel(self.channel * mask.mask)


class BioMedicalImage:
    def __init__(self, img: np.ndarray, img_params: ImageParameter, segmentation: BioMedicalInstanceSegmentation = None):
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
