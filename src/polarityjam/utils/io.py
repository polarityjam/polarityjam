"""Input output functions for the project."""
import glob
import os
import shutil
import zipfile
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import skimage.io
import yaml  # type: ignore

from polarityjam.polarityjam_logging import get_logger

# Global variable to save program call time.
CALL_TIME = None


def read_image(filename: Union[Path, str]) -> np.ndarray:
    """Read an RGB or grayscale image with the scikit-learn library. Swaps axis if channels are not on last position.

    Args:
        filename:
            Path to the image file.

    Returns:
        image with channels on last position.
    """
    img_ = skimage.io.imread(filename)

    if len(img_.shape) <= 2:
        img_ = np.array([img_, img_])

    if img_.shape[0] < min(img_.shape[1], img_.shape[2]):
        get_logger().warning("Image has channels on first position. Swapping axis.")
        img = np.swapaxes(np.swapaxes(img_, 0, 2), 0, 1)
    else:
        img = img_

    return img


def write_dict_to_yml(yml_file: Union[str, Path], d: dict) -> bool:
    """Write a dictionary to a file in yml format.

    Args:
        yml_file:
            Path to the yml file.
        d:
            Dictionary to be written to the yml file.

    Returns:
        True if successful.

    """
    yml_file = Path(yml_file)
    p = Path(yml_file.parent)
    p.mkdir(parents=True, exist_ok=True)

    with open(yml_file, "w+") as yml_f:
        yml_f.write(yaml.dump(d, Dumper=yaml.Dumper))

    return True


def get_dict_from_yml(yml_file: Path) -> dict:
    """Read a dictionary from a file in yml format."""
    with open(str(yml_file)) as yml_f:
        d = yaml.safe_load(yml_f)

    if not isinstance(d, dict):
        raise TypeError("Yaml file %s invalid!" % str(yml_file))

    return d


def create_path_recursively(path: Union[str, Path]) -> bool:
    """Create a path. Creates missing parent folders.

    Args:
        path:
            Path to be created.

    Returns:
        True if successful.

    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)

    return True


def get_tif_list(path: Union[str, Path]) -> List[str]:
    """Get a list of all tif files in a folder.

    Args:
        path:
            Path to the folder.

    Returns:
        List of all tif files in the folder.

    """
    path = str(path)
    if not path.endswith(os.path.sep):
        path = path + os.path.sep

    return glob.glob(path + "*.tif")


def read_key_file(path: Union[str, Path]) -> pd.DataFrame:
    """Read the key file and return a pandas dataframe.

    Args:
        path:
            Path to the key file.

    Returns:
        Pandas dataframe with the key file.

    """
    return pd.read_csv(path)


def list_files_recursively(
    path: Path,
    root: Optional[Union[Path]] = None,
    relative: bool = False,
    endswith: Optional[str] = None,
    recursive: bool = True,
) -> List[Path]:
    """List all files in a repository recursively."""
    if not root:
        root = path
    files_list = []

    for cur_root, dirs, files in os.walk(str(path)):
        cur_root = str(Path(cur_root))

        for d in dirs:
            if recursive:
                files_list += list_files_recursively(
                    Path(cur_root).joinpath(d), root, relative, endswith
                )
        for fi in files:
            if endswith:
                if not fi.endswith(endswith):
                    continue
            if relative:
                files_list.append(Path(cur_root).joinpath(fi).relative_to(root))
            else:
                files_list.append(Path(cur_root).joinpath(fi))
        break

    return files_list


def copy(file: Union[str, Path], path_to: Union[str, Path]) -> Path:
    """Copy a file A to either folder B or file B. Makes sure folder structure for target exists.

    Args:
        file:
            Path to the file to be copied.
        path_to:
            Path to the target folder or file.

    Returns:
        Path to the copied file.

    """
    file = Path(file)
    path_to = Path(path_to)

    if os.path.exists(path_to) and os.path.samefile(file, path_to):
        return path_to

    create_path_recursively(path_to.parent)

    return Path(shutil.copy(file, path_to))


def check_zip(path):
    """Check a given zip file."""
    return zipfile.is_zipfile(path)
