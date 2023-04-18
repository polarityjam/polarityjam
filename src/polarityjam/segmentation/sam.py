"""Module holding the SAM segmentation model."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from polarityjam.controller.segmenter import Segmenter
from polarityjam.model.parameter import ImageParameter, SegmentationParameter
from polarityjam.polarityjam_logging import get_logger
from polarityjam.utils.url import download_resource


class SamSegmenter(Segmenter):
    """SAM segmentation model."""

    MODEL_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    MODEL_URL_L = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
    DOWNLOAD_PATH_REL = Path(os.path.dirname(os.path.realpath(__file__))).joinpath(
        "SAM", "sam_vit_h_4b8939.pth"
    )

    def __init__(self, params: SegmentationParameter):
        """Initialize the segmenter with the given parameters."""
        self.params = params
        self.model_path = None

    def segment(
        self, img: np.ndarray, path: Optional[str] = None, mode: Optional[str] = None
    ) -> np.ndarray:
        """Segment the given image.

        Args:
            img:
                The image to segment.
            path:
                The path to the image.
            mode:
                The mode to use for segmentation.

        Returns:
            The segmented image as numpy array.

        """
        if mode == "nucleus":
            return self._segment_single_channel(img[:, :, 1])
        elif mode == "cell":
            return self._segment_single_channel(img[:, :, 0])
        elif mode == "organelle":
            return self._segment_single_channel(img[:, :, 2])
        else:
            raise ValueError("Mode must be either nucleus, organelle or cell.")

    def _segment_single_channel(self, img: np.ndarray) -> np.ndarray:
        img_e = np.expand_dims(img, axis=-1)
        img_3d = np.repeat(img_e, 3, axis=-1).astype(np.uint8)

        sam = sam_model_registry["vit_h"](checkpoint=str(self.DOWNLOAD_PATH_REL))
        mask_generator = SamAutomaticMaskGenerator(sam)
        masks = mask_generator.generate(img_3d)
        return self._to_instance_segmentation(masks)

    @staticmethod
    def _to_instance_segmentation(masks: List[Dict[str, Any]]) -> np.ndarray:
        h = np.zeros(masks[0]["segmentation"].shape)

        for idx, i in enumerate(masks):
            if idx == 0:
                continue  # background class

            f = i["segmentation"].astype(np.uint8)
            f = f * idx
            h = np.where(f > 0, f, h)

        return h.astype(np.uint8)

    def prepare(
        self, img: np.ndarray, img_parameter: ImageParameter
    ) -> Tuple[Optional[np.ndarray], ImageParameter]:
        """Prepare the image for segmentation.

        Args:
            img:
                The image to prepare.
            img_parameter:
                The image parameter.

        Returns:
            Tuple of the prepared image and the image parameter.

        """
        if not self.DOWNLOAD_PATH_REL.exists():
            download_resource(self.MODEL_URL, self.DOWNLOAD_PATH_REL)

        params_prep_img = ImageParameter()
        px_to_m_r = img_parameter.pixel_to_micron_ratio
        params_prep_img.reset()
        params_prep_img.pixel_to_micron_ratio = px_to_m_r

        numpy_img = np.zeros([img.shape[0], img.shape[1], 3])

        if img_parameter.channel_junction < 0:
            raise ValueError("No junction channel found.")
        else:
            get_logger().info(
                "Junction channel used for segmentation at position: %s"
                % str(img_parameter.channel_junction)
            )
            im_junction = img[:, :, img_parameter.channel_junction]
            params_prep_img.channel_junction = 0

            numpy_img[:, :, 0] = im_junction

        if img_parameter.channel_nucleus >= 0:
            get_logger().info(
                "Nucleus channel used for segmentation at position: %s"
                % str(img_parameter.channel_nucleus)
            )
            im_nucleus = img[:, :, img_parameter.channel_nucleus]
            params_prep_img.channel_nucleus = 1

            numpy_img[:, :, 1] = im_nucleus

        if img_parameter.channel_organelle >= 0:
            get_logger().info(
                "Organelle channel used for segmentation at position: %s"
                % str(img_parameter.channel_organelle)
            )
            im_organelle = img[:, :, img_parameter.channel_organelle]
            params_prep_img.channel_organelle = 2

            numpy_img[:, :, 2] = im_organelle

        return numpy_img, params_prep_img
