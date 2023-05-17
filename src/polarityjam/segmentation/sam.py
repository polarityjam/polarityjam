"""Module holding the SAM segmentation model."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from polarityjam.controller.segmenter import SegmentationMode, Segmenter
from polarityjam.model.parameter import ImageParameter, SegmentationParameter
from polarityjam.polarityjam_logging import get_logger
from polarityjam.utils.url import download_resource


class SamSegmenter(Segmenter):
    """SAM segmentation model."""

    DOWNLOAD_PATH_REL = Path(os.path.dirname(os.path.realpath(__file__))).joinpath(
        "SAM"
    )

    def __init__(self, params: SegmentationParameter):
        """Initialize the segmenter with the given parameters."""
        self.params = params
        self.model_url = params.model_url  # type: ignore
        self.model_path = self.DOWNLOAD_PATH_REL.joinpath(
            os.path.split(self.model_url)[-1]
        )

    def segment(
        self,
        img: np.ndarray,
        path: Optional[str] = None,
        mode: Optional[SegmentationMode] = SegmentationMode.CELL,
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
        if mode == SegmentationMode.NUCLEUS:
            return self._segment_single_channel(img[:, :, 1])
        elif mode == SegmentationMode.CELL:
            return self._segment_single_channel(img[:, :, 0])
        elif mode == SegmentationMode.ORGANELLE:
            return self._segment_single_channel(img[:, :, 2])
        else:
            raise ValueError("Mode must be either nucleus, organelle or cell.")

    def _segment_single_channel(self, img: np.ndarray) -> np.ndarray:
        img_e = np.expand_dims(img, axis=-1)
        img_3d = np.repeat(img_e, 3, axis=-1).astype(np.uint8)

        sam = sam_model_registry["vit_h"](checkpoint=str(self.model_path))
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
        if not self.model_path.exists():
            download_resource(self.model_url, self.model_path)

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
