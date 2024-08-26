"""Module holding the Cellpose segmentation model."""
from pathlib import Path
from typing import Optional, Tuple, Union

import cellpose.models
import numpy as np

from polarityjam import ImageParameter
from polarityjam.controller.segmenter import SegmentationMode, Segmenter
from polarityjam.polarityjam_logging import get_logger


class CellposeSegmenter(Segmenter):
    """Cellpose segmentation class."""

    def __init__(self, params):
        """Initialize the object with the given parameters."""
        super().__init__(params)
        self.params = params

    def _check_params(self, path):
        if self.params.use_given_mask:
            if path is None:
                raise ValueError(
                    "Path to segmentation mask must be given if use_given_mask is True."
                )

        if self.params.store_segmentation:
            if path is None:
                raise ValueError(
                    "Path to segmentation mask must be given if store_segmentation is True."
                )

    def segment(
        self,
        img: np.ndarray,
        path: Optional[Union[Path, str]] = None,
        mode: Optional[SegmentationMode] = SegmentationMode.CELL,
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
        self._check_params(path)
        if mode is None:
            mode = SegmentationMode.CELL

        if isinstance(mode, str):
            try:
                mode = SegmentationMode(mode)
            except ValueError:
                raise ValueError(
                    'Mode must be either "nucleus", "organelle", "cell" or "junction".'
                )

        if mode == SegmentationMode.JUNCTION:
            raise ValueError("This segmentation algorithm does not support this mode!")
        elif mode == SegmentationMode.ORGANELLE:
            get_logger().info(
                "This model is probably not trained for organelles segmentation. Please handle results with care."
            )
            return self._load_or_get_cellpose_segmentation(img, path, True)
        elif mode == SegmentationMode.CELL:
            get_logger().info("Start segmentation procedure for cells...")
            return self._load_or_get_cellpose_segmentation(img, path, True)
        elif mode == SegmentationMode.NUCLEUS:
            get_logger().info("Start segmentation procedure for nuclei...")
            return self._load_or_get_cellpose_segmentation(img, path, False)
        else:
            raise ValueError(
                'Mode must be either "nucleus", "organelle", "cell" or "junction".'
            )

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

        # check which channel is configured to use for cell segmentation:
        channel_cell_segmentation = img_parameter.channel_junction
        if self.params.channel_cell_segmentation != "":
            try:
                channel_cell_segmentation = img_parameter.__getattribute__(
                    self.params.channel_cell_segmentation
                )
            except AttributeError:
                get_logger().error(
                    "Channel %s does not exist! Wrong segmentation configuration!"
                    % self.params.channel_cell_segmentation
                )

        # check which channel is configured to use for nuclei segmentation:
        channel_nuclei_segmentation = img_parameter.channel_nucleus
        if self.params.channel_nuclei_segmentation != "":
            try:
                channel_nuclei_segmentation = img_parameter.__getattribute__(
                    self.params.channel_nuclei_segmentation
                )
            except AttributeError:
                get_logger().error(
                    "Channel %s does not exist! Wrong segmentation configuration!"
                    % self.params.channel_nuclei_segmentation
                )

        if channel_cell_segmentation >= 0:
            get_logger().info(
                "Channel that will be used for cell segmentation at position: %s"
                % str(channel_cell_segmentation)
            )
            im_junction = img[:, :, channel_cell_segmentation]
            params_prep_img.channel_junction = (
                0  # might be called junction channel, but content depends on config
            )

        if channel_nuclei_segmentation >= 0:
            get_logger().info(
                "Channel that will be used for nucleus segmentation at position: %s"
                % str(channel_nuclei_segmentation)
            )
            im_nucleus = img[:, :, channel_nuclei_segmentation]
            params_prep_img.channel_nucleus = (
                1  # might be called nuclei channel, but content depends on config
            )

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
            else self.params.estimated_nucleus_diameter
        )
        get_logger().info(
            "Calculate cellpose %s segmentation. This might take some time..."
            % seg_type
        )
        get_logger().info(
            "Using model type '%s' with estimated cell diameter %s, cellprob_threshold %s,"
            " flow threshold %s, and number of iterations %s."
            % (
                model_type,
                estimated_cell_diameter,
                self.params.cellprob_threshold,
                self.params.flow_threshold,
                self.params.niter,
            )
        )
        model = self._get_cellpose_model(cells)
        if im_seg.ndim > 2:
            channels = [1, 2]
            if not cells:
                channels = [0, 0]
                im_seg = im_seg[1, :, :]
        else:
            channels = [0, 0]

        if self.params.model_type == "custom":
            masks, flows, styles = model.eval(  #
                im_seg,
                diameter=estimated_cell_diameter,
                flow_threshold=self.params.flow_threshold,
                cellprob_threshold=self.params.cellprob_threshold,
                niter=self.params.niter,
                channels=channels,
            )
        else:
            masks, flows, styles, diams = model.eval(
                im_seg,
                diameter=estimated_cell_diameter,
                flow_threshold=self.params.flow_threshold,
                cellprob_threshold=self.params.cellprob_threshold,
                niter=self.params.niter,
                channels=channels,
            )

        if self.params.store_segmentation:
            segmentation_list = {"masks": masks}
            segmentation, _ = self._get_segmentation_file_name(filepath, cells)

            get_logger().info("Storing cellpose segmentation: %s" % segmentation)
            np.save(str(segmentation), segmentation_list, allow_pickle=True)

        return masks

    def _get_segmentation_file_name(self, filepath, cells=True):
        segmentation = None
        stem = None
        if self.params.use_given_mask:
            stem = Path(filepath).stem

            suffix = "_seg.npy" if cells else "_seg_nuc.npy"

            if self.params.manually_annotated_mask:
                suffix = self.params.manually_annotated_mask
            segmentation = Path(filepath).parent.joinpath(stem + suffix)

        return segmentation, stem

    def _load_or_get_cellpose_segmentation(self, img_seg, filepath, cells=True):
        get_logger().info("Look up cellpose segmentation on disk...")
        segmentation, _ = self._get_segmentation_file_name(filepath, cells)

        if self.params.use_given_mask and segmentation.exists():
            get_logger().info("Load existing segmentation from %s ..." % segmentation)

            # in case an annotated mask is available
            cellpose_seg = np.load(str(segmentation), allow_pickle=True)
            cellpose_mask = cellpose_seg.item()["masks"]
        else:
            cellpose_mask = self._get_cellpose_segmentation(img_seg, filepath, cells)

        get_logger().info(
            "Detected number of cellpose labels: %s" % len(np.unique(cellpose_mask))
        )

        get_logger().info(
            "Cellpose segmentation for %s done." % ("cells" if cells else "nuclei")
        )

        return cellpose_mask
