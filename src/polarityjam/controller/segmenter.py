"""Segmentation class base and CellposeSegmentation class."""
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Optional, Tuple, Union

import cellpose.models
import numpy as np
import skimage
from skimage import morphology

from polarityjam.model.parameter import ImageParameter, SegmentationParameter
from polarityjam.polarityjam_logging import get_logger


class Segmenter:
    """Abstract class for an object performing a segmentation procedure."""

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, params: SegmentationParameter):
        """Initialize the segmenter with the given parameters."""
        self.params = params

    @abstractmethod
    def segment(
        self, img: np.ndarray, path: Optional[str] = None, mode: Optional[str] = None
    ) -> np.ndarray:
        """Perform segmentation and return a mask image.

        Path can point to a model to load/state index/parameter
        file or something else needed to load a checkpoint needed for segmentation.

        Args:
            img:
                The image prepared for segmentation.
            path:
                Path to a model to load/state index/parameter file or smth. else needed to load a checkpoint needed
                to perform the segmentation.
            mode:
                The mode of the segmentation. Could point to different models or something else.

        Returns:
            A mask as np.ndarray image.

        """
        raise NotImplementedError

    @abstractmethod
    def prepare(
        self, img: np.ndarray, input_parameter: ImageParameter
    ) -> Tuple[Optional[np.ndarray], ImageParameter]:
        """Perform preparation for a given image to be able to do the segmentation.

        Should return prepared image
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
    """Cellpose segmentation class."""

    def __init__(self, params: SegmentationParameter):
        """Initialize the object with the given parameters."""
        super().__init__(params)
        self.params = params

    def segment(
        self,
        img: np.ndarray,
        path: Optional[Union[Path, str]] = None,
        mode: Optional[str] = None,
    ) -> np.ndarray:
        """Perform the segmentation of the given image.

        If a path is given, the model is loaded from the given path.

        Args:
            img:
                The image prepared for segmentation.
            path:
                Path to a model to load/state index/parameter file or smth. else needed to load a checkpoint needed
            mode:
                If mode is "nucleus", indicates that the nucleus channel should be segmented.

        Returns:
            A mask as np.ndarray image.

        """
        cells = False if mode == "nucleus" else True
        return self._load_or_get_cellpose_segmentation(img, path, cells)

    def prepare(
        self, img: np.ndarray, img_parameter: ImageParameter
    ) -> Tuple[Optional[np.ndarray], ImageParameter]:
        """Prepare the image for segmentation.

        Returns an image that has the junction channel first, then the nucleus
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
            get_logger().info(
                "Junction channel at position: %s" % str(img_parameter.channel_junction)
            )
            im_junction = img[:, :, img_parameter.channel_junction]
            params_prep_img.channel_junction = 0

        if img_parameter.channel_nucleus >= 0:
            get_logger().info(
                "Nucleus channel at position: %s" % str(img_parameter.channel_nucleus)
            )
            im_nucleus = img[:, :, img_parameter.channel_nucleus]
            params_prep_img.channel_nucleus = 1

        if im_nucleus is not None:
            return np.array([im_junction, im_nucleus]), params_prep_img
        else:
            return im_junction, params_prep_img

    def _get_cellpose_model(self, cells=True):
        """Get the specified cellpose model."""
        if self.params.model_type == "custom":
            get_logger().info("Loading custom model from: %s" % self.params.model_path)
            model = cellpose.models.CellposeModel(
                gpu=self.params.use_gpu, pretrained_model=self.params.model_path
            )
        else:
            model_type = (
                self.params.model_type if cells else self.params.model_type_nucleus
            )
            model = cellpose.models.Cellpose(
                gpu=self.params.use_gpu, model_type=model_type
            )

        return model

    def _get_cellpose_segmentation(self, im_seg, filepath, cells=True):
        """Get the cellpose segmentation.

        Expects im_seg to have junction channel first, then nucleus channel.

        """
        seg_type = "cell" if cells else "nuclei"
        model_type = self.params.model_type if cells else self.params.model_type_nucleus
        estimated_cell_diameter = (
            self.params.estimated_cell_diameter
            if cells
            else self.params.estimated_cell_diameter_nucleus
        )
        get_logger().info(
            "Calculate cellpose %s segmentation. This might take some time..."
            % seg_type
        )
        get_logger().info(
            "Using model type '%s' with estimated cell diameter %s, cellprob_threshold %s and flow threshold %s"
            % (
                model_type,
                estimated_cell_diameter,
                self.params.cellprob_threshold,
                self.params.flow_threshold,
            )
        )
        model = self._get_cellpose_model(cells)
        if im_seg.ndim > 1:
            channels = [1, 2]
        else:
            channels = [0, 0]

        if self.params.model_type == "custom":
            masks, flows, styles = model.eval(
                im_seg,
                diameter=estimated_cell_diameter,
                flow_threshold=self.params.flow_threshold,
                cellprob_threshold=self.params.cellprob_threshold,
                channels=channels,
            )
        else:
            masks, flows, styles, diams = model.eval(
                im_seg,
                diameter=estimated_cell_diameter,
                flow_threshold=self.params.flow_threshold,
                cellprob_threshold=self.params.cellprob_threshold,
                channels=channels,
            )

        if self.params.store_segmentation:
            segmentation_list = {"masks": masks}
            segmentation, _ = self._get_segmentation_file_name(filepath, cells)

            get_logger().info("Storing cellpose segmentation: %s" % segmentation)
            np.save(str(segmentation), segmentation_list, allow_pickle=True)

        return masks

    def _get_segmentation_file_name(self, filepath, cells=True):
        stem = Path(filepath).stem

        suffix = "_seg.npy" if cells else "_seg_nuc.npy"

        if self.params.manually_annotated_mask:
            suffix = self.params.manually_annotated_mask
        segmentation = Path(filepath).parent.joinpath(stem + suffix)

        return segmentation, stem

    def _load_or_get_cellpose_segmentation(self, img_seg, filepath, cells=True):
        get_logger().info("Look up cellpose segmentation...")
        segmentation, _ = self._get_segmentation_file_name(filepath, cells)

        if segmentation.exists() and self.params.use_given_mask:
            get_logger().info("Load existing segmentation from %s ..." % segmentation)

            # in case an annotated mask is available
            cellpose_seg = np.load(str(segmentation), allow_pickle=True)
            cellpose_mask = cellpose_seg.item()["masks"]
        else:
            cellpose_mask = self._get_cellpose_segmentation(img_seg, filepath, cells)

        if self.params.clear_border:
            cellpose_mask_clear_border = skimage.segmentation.clear_border(
                cellpose_mask
            )
            number_of_cellpose_borders = len(np.unique(cellpose_mask)) - len(
                np.unique(cellpose_mask_clear_border)
            )
            cellpose_mask = cellpose_mask_clear_border

            get_logger().info(
                "Removed number of cellpose borders: %s" % number_of_cellpose_borders
            )

        if self.params.remove_small_objects:
            cellpose_mask_remove_small_objects = morphology.remove_small_objects(
                cellpose_mask, self.params.min_cell_size, connectivity=2
            )
            number_of_cellpose_small_objects = len(np.unique(cellpose_mask)) - len(
                np.unique(cellpose_mask_remove_small_objects)
            )
            cellpose_mask = cellpose_mask_remove_small_objects

            get_logger().info(
                "Removed number of small objects: %s" % number_of_cellpose_small_objects
            )

        get_logger().info(
            "Detected number of cellpose labels: %s" % len(np.unique(cellpose_mask))
        )

        return cellpose_mask
