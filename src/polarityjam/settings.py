"""General settings for the project."""
from enum import Enum
from pathlib import Path


class Settings(Enum):
    """Settings for the program."""

    # model_path
    dynamic_loading_prefix = "polarityjam.segmentation"
    segmentation_algorithm = "segmentation_algorithm"
    segmentation_algorithm_default = "CellposeSegmenter"

    # installation base
    installation_base = str(Path.home().joinpath(".polarityjam", "collection"))
    model_base = str(Path.home().joinpath(".polarityjam", "model"))
