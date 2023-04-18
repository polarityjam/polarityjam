"""Segmentation class base and CellposeSegmentation class."""
from abc import ABCMeta, abstractmethod
from pydoc import locate
from typing import Optional, Tuple

import numpy as np

from polarityjam.model.parameter import (
    ImageParameter,
    RuntimeParameter,
    SegmentationAlgorithmE,
    SegmentationParameter,
)


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


def load_segmenter(params_runtime: RuntimeParameter, parameter: Optional[dict] = None):
    """Load a segmenter based on the given parameters.

    Args:
        params_runtime:
            The runtime parameters. Must contain the segmentation algorithm attribute.
        parameter:
            The parameters for the segmenter as a dictionary. If None, the default parameters are used.
            Unnecessary parameters are ignored.

    Returns:
        A tuple of the Segmeter object and its parameters as SegmentationParameter object.
    """
    m = locate(SegmentationAlgorithmE[params_runtime.segmentation_algorithm].value)  # type: ignore
    assert isinstance(m, type), "Segmentation algorithm not found!"
    segmentation_parameter = SegmentationParameter(
        params_runtime.segmentation_algorithm, parameter
    )
    segmentor = m(segmentation_parameter)  # type: ignore
    return segmentor, segmentation_parameter  # type: ignore
