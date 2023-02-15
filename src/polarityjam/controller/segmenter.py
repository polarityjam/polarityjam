from pathlib import Path
from typing import Tuple, Union

import cellpose.models
import numpy as np
import skimage
from skimage import morphology

from polarityjam.model.parameter import SegmentationParameter, ImageParameter
from polarityjam.polarityjam_logging import get_logger

from abc import ABCMeta, abstractmethod


class Segmenter:
    """Abstract class for an object performing a segmentation procedure."""
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, params: SegmentationParameter):
        self.params = params

    @abstractmethod
    def segment(self, img: np.ndarray, path: str=None) -> np.ndarray:
        """Should perform segmentation and return a mask image. Path can point to a model to load/state index/parameter
        file or something else needed to load a checkpoint needed for segmentation.

        Args:
            img:
                The image prepared for segmentation.

            path:
                Path to a model to load/state index/parameter file or smth. else needed to load a checkpoint needed
                to perform the segmentation.

        Returns:
            A mask as np.ndarray image.

        """
        raise NotImplementedError

    @abstractmethod
    def prepare(self, img: np.ndarray, input_parameter: ImageParameter) -> Tuple[np.ndarray, ImageParameter]:
        """Should perform preparation for a given image to perform the segmentation. Should return prepared image
        and its parameters. This could be a resizing, cropping, selecting channels, etc. E.g. whatever is needed to
        perform the segmentation.

        Args:
            img:
                The input image to prepare for segmentation as a numpy array.
            input_parameter:
                The parameters of the input image

        Returns:
            A tuple of the prepared image and its parameters as ImageParameter object.

        """
        raise NotImplementedError


class CellposeSegmenter(Segmenter):
    """Cellpose segmentation class"""

    def __init__(self, params: SegmentationParameter):
        super().__init__(params)
        self.params = params

    def segment(self, img: np.ndarray, path: Union[Path, str]=None) -> np.ndarray:
        """Performs the segmentation of the given image. If a path is given, the model is loaded from the given path.

        Args:
            img:
                The image prepared for segmentation.
            path:
                Path to a model to load/state index/parameter file or smth. else needed to load a checkpoint needed

        Returns:
            A mask as np.ndarray image.

        """
        return self._load_or_get_cellpose_segmentation(img, path)

    def prepare(self, img: np.ndarray, img_parameter: ImageParameter) -> Tuple[np.ndarray, ImageParameter]:
        """Prepares the image for segmentation. Returns an image that has the junction channel first, then the nucleus
        channel and the last channel is the cytoplasm channel.

        Args:
            img:
                The input image to prepare for segmentation as a numpy array.
            img_parameter:
                The parameters of the input image

        Returns:
            A tuple of the prepared image and its parameters as ImageParameter object.

        """

        get_logger().info("Image shape: %s" % str(img.shape))

        params_prep_img = ImageParameter()
        px_to_m_r = img_parameter.pixel_to_micron_ratio
        params_prep_img.reset()
        params_prep_img.pixel_to_micron_ratio = px_to_m_r

        im_junction = None
        im_nucleus = None

        if img_parameter.channel_junction >= 0:
            get_logger().info("Junction channel at position: %s" % str(img_parameter.channel_junction))
            im_junction = img[:, :, img_parameter.channel_junction]
            params_prep_img.channel_junction = 0

        if img_parameter.channel_nucleus >= 0:
            get_logger().info("Nucleus channel at position: %s" % str(img_parameter.channel_nucleus))
            im_nucleus = img[:, :, img_parameter.channel_nucleus]
            params_prep_img.channel_nucleus = 1

        if im_nucleus is not None:
            return np.array([im_junction, im_nucleus]), params_prep_img
        else:
            return im_junction, params_prep_img

    def _get_cellpose_model(self):
        """Gets the specified cellpose model"""

        if self.params.model_type == "custom":
            get_logger().info("Loading custom model from: %s" % self.params.model_path)
            model = cellpose.models.CellposeModel(gpu=self.params.use_gpu, pretrained_model=self.params.model_path)
        else:
            model = cellpose.models.Cellpose(gpu=self.params.use_gpu, model_type=self.params.model_type)

        return model

    def _get_cellpose_segmentation(self, im_seg, filepath):
        """Gets the cellpose segmentation. Expects im_seg to have junction channel first, then nucleus channel."""
        get_logger().info("Calculate cellpose segmentation. This might take some time...")
        get_logger().info("Using model type '%s' with estimated cell diameter %s, cellprob_threshold %s and flow threshold %s" % (
        self.params.model_type, self.params.estimated_cell_diameter, self.params.cellprob_threshold, self.params.flow_threshold))
        model = self._get_cellpose_model()
        if im_seg.ndim > 1:
            channels = [1, 2]
        else:
            channels = [0, 0]

        # masks, flows, styles, diams = model.eval(im_seg, channels=channels)

        if self.params.model_type == "custom":
            masks, flows, styles = model.eval(im_seg, diameter=self.params.estimated_cell_diameter,
                                              flow_threshold=self.params.flow_threshold,
                                              cellprob_threshold=self.params.cellprob_threshold,
                                              channels=channels)
        else:
            masks, flows, styles, diams = model.eval(im_seg, diameter=self.params.estimated_cell_diameter,
                                                     flow_threshold=self.params.flow_threshold,
                                                     cellprob_threshold = self.params.cellprob_threshold,
                                                     channels=channels)

        if self.params.store_segmentation:
            segmentation_list = {"masks": masks}
            segmentation, _ = self._get_segmentation_file_name(filepath)

            get_logger().info("Storing cellpose segmentation: %s" % segmentation)
            np.save(str(segmentation), segmentation_list, allow_pickle=True)

        return masks

    def _get_segmentation_file_name(self, filepath):
        stem = Path(filepath).stem

        suffix = "_seg.npy"
        if self.params.manually_annotated_mask:
            suffix = self.params.manually_annotated_mask
        segmentation = Path(filepath).parent.joinpath(stem + suffix)

        return segmentation, stem

    def _load_or_get_cellpose_segmentation(self, img_seg, filepath):
        get_logger().info("Look up cellpose segmentation...")
        segmentation, _ = self._get_segmentation_file_name(filepath)

        if segmentation.exists() and self.params.use_given_mask:
            get_logger().info("Load existing segmentation from %s ..." % segmentation)

            # in case an annotated mask is available
            cellpose_seg = np.load(str(segmentation), allow_pickle=True)
            cellpose_mask = cellpose_seg.item()['masks']

        else:
            cellpose_mask = self._get_cellpose_segmentation(img_seg, filepath)

        if self.params.clear_border:
            cellpose_mask_clear_border = skimage.segmentation.clear_border(cellpose_mask)
            number_of_cellpose_borders = len(np.unique(cellpose_mask)) - len(np.unique(cellpose_mask_clear_border))
            cellpose_mask = cellpose_mask_clear_border

            get_logger().info("Removed number of cellpose borders: %s" % number_of_cellpose_borders)

            cellpose_mask_remove_small_objects = morphology.remove_small_objects(
                cellpose_mask, self.params.min_cell_size, connectivity=2
            )
            number_of_cellpose_small_objects = len(np.unique(cellpose_mask)) - len(
                np.unique(cellpose_mask_remove_small_objects))
            cellpose_mask = cellpose_mask_remove_small_objects

            get_logger().info("Removed number of small objects: %s" % number_of_cellpose_small_objects)

        get_logger().info("Detected number of cellpose labels: %s" % len(np.unique(cellpose_mask)))

        return cellpose_mask
