"""General settings for the project."""
from enum import Enum


class Settings(Enum):
    """Settings for the program."""

    # model_path
    dynamic_loading_prefix = "polarityjam.segmentation"
    segmentation_algorithm = "segmentation_algorithm"
    segmentation_algorithm_default = "CellposeSegmenter"
