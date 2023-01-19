import glob
import os
import shutil
import time
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
import skimage.io
import yaml

"""
Global variable to save program call time.
"""

CALL_TIME = None


def read_parameters(parameter_file: str) -> dict:
    """Reads in default parameters and replaces user defined parameters.

    Args:
        parameter_file:
            Path to the parameter file.
    Returns:
        dictionary where all missing parameter values are replaced with the default values.

    """
    current_path = Path(os.path.dirname(os.path.realpath(__file__)))

    param_base_file = Path(current_path).joinpath("resources", "parameters.yml")

    with open(param_base_file, 'r') as yml_f:
        parameters = yaml.safe_load(yml_f)

    with open(parameter_file) as file:
        parameters_local = yaml.safe_load(file)

    # overwrite global parameters with local setting
    for key in parameters_local:
        parameters[key] = parameters_local[key]

    return parameters


def read_image(filename: Union[Path, str]) -> np.ndarray:
    """Reads an RGB or grayscale image with the scikit-learn library. Swaps axis if channels are not on last position.

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
        print("Warning: channel is on the first dimension of the image.")
        img = np.swapaxes(np.swapaxes(img_, 0, 2), 0, 1)
    else:
        img = img_

    return img


def write_dict_to_yml(yml_file: Union[str, Path], d: dict) -> bool:
    """Writes a dictionary to a file in yml format.

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

    with open(yml_file, 'w+') as yml_f:
        yml_f.write(yaml.dump(d, Dumper=yaml.Dumper))

    return True


def create_path_recursively(path: Union[str, Path]) -> bool:
    """Creates a path. Creates missing parent folders.

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


def list_files_recursively(path: Union[str, Path], root: Union[str, Path] = None, relative: bool = False) -> List[Path]:
    """List all files in a folder recursively.

    Args:
        path:
            Path to the folder.
        root:
            Root path to be removed from the file path.
        relative:
            If True, the root path is removed from the file path.

    Returns:
        List of all files in the folder.

    """
    path_ = Path(path)
    if not root:
        root = path_
    files_list = []

    for cur_root, dirs, files in os.walk(path_):
        cur_root = Path(cur_root)

        for d in dirs:
            files_list += list_files_recursively(cur_root.joinpath(d), root, relative)
        for fi in files:
            if relative:
                files_list.append(cur_root.joinpath(fi).relative_to(root))
            else:
                files_list.append(cur_root.joinpath(fi))
        break

    return files_list


def get_doc_file_prefix() -> str:
    """Get the time when the program was called.

    Returns:
        Time when the program was called.

    """
    global CALL_TIME

    if not CALL_TIME:
        CALL_TIME = time.strftime('%Y%m%d_%H-%M-%S')

    call_time = CALL_TIME

    return "run_%s" % call_time


def copy(file: Union[str, Path], path_to: Union[str, Path]) -> Path:
    """Copies a file A to either folder B or file B. Makes sure folder structure for target exists.

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
