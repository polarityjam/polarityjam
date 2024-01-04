"""Normalization functions for images and arrays."""
from typing import List, Optional, Tuple, Union

import numpy as np


def normalize_img_channel_wise(
    x: np.ndarray,
    source_limits: Optional[
        Union[
            List[Tuple[Union[int, float], Union[int, float]]],
            Tuple[Union[int, float], Union[int, float]],
        ]
    ] = None,
    target_limits: Optional[
        Union[
            List[Tuple[Union[int, float], Union[int, float]]],
            Tuple[Union[int, float], Union[int, float]],
        ]
    ] = None,
):
    """Normalize the given image channel-wise. Each channel is assumed to be grayscale.

    Args:
        x:
            numpy array of shape (height, width, num_channels). Each channel is assumed to be grayscale.
        source_limits:
            The source limits for each channel. If None, the min and max of each channel is used.
            If a list of tuples, each tuple is assumed to be the source limits for each channel.
        target_limits:
            The target limits for each channel. If None, (0, 1) is used.
            If a list of tuples, each tuple is assumed to be the target limits for each channel.

    Returns:
        The channel normalized image

    """
    num_channels = x.shape[-1]

    s_lims = [None] * num_channels
    if source_limits is not None:
        s_lims = (
            source_limits  # type: ignore
            if isinstance(source_limits, list)
            else [source_limits] * num_channels  # type: ignore
        )

    t_lims = [None] * num_channels
    if target_limits is not None:
        t_lims = (
            target_limits  # type: ignore
            if isinstance(target_limits, list)
            else [target_limits] * num_channels  # type: ignore
        )

    assert (
        len(s_lims) == len(t_lims) == num_channels
    ), "Number of source and target limits must match number of channels"

    for idx, _ in enumerate(range(num_channels)):
        x[:, :, idx] = normalize_arr(x[:, :, idx], s_lims[idx], t_lims[idx])

    return x


def normalize_arr(
    x: np.ndarray,
    source_limits: Optional[Tuple[Union[int, float], Union[int, float]]] = None,
    target_limits: Optional[Tuple[Union[int, float], Union[int, float]]] = None,
) -> np.ndarray:
    """Normalize the given array.

    Args:
        x:
            numpy array of shape (height, width)
        source_limits:
            The source limits. If None, the min and max of the array is used.
        target_limits:
            The target limits. If None, (0, 1) is used.

    Returns:
        The normalized array
    """
    if source_limits is None:
        source_limits = (x.min(initial=0), x.max(initial=np.iinfo(x.dtype).max))

    if target_limits is None:
        target_limits = (0, 1)

    if source_limits[0] == source_limits[1] or target_limits[0] == target_limits[1]:
        return x * 0
    else:
        x_std = (x - source_limits[0]) / (source_limits[1] - source_limits[0])
        x_scaled = x_std * (target_limits[1] - target_limits[0]) + target_limits[0]
        return x_scaled
