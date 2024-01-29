"""Module for support computing statistics."""
import numpy as np
import pandas as pd

from polarityjam.polarityjam_logging import get_logger


def compute_polarity_index(
    angles: np.ndarray,
    cue_direction: float = 0.0,
    stats_mode: str = "directional",
    unit: str = "radians",
):
    """Compute the polarity index of a list of angles.

    Args:
        angles:
            array with directional or axial data
        cue_direction:
            direction of the imposed cue, used for V-score computation
        stats_mode:
            'directional' or 'axial'
        unit:
            'degrees' or 'radians'

    Returns:
        list of values [alpha_m, R, c], alpha_m is the mean direction (in degrees),
        R is the resultant vector length, c is a measure of deviation from the mean angle,
        used to calculate the V-score: V=cR

    """
    get_logger().info("Compute circular statistics")

    sum_cos = 0.0
    sum_sin = 0.0

    p = 1.0
    if stats_mode == "axial":
        p = 2.0

    if unit == "degrees":
        angles = np.deg2rad(angles)

    for angle in angles:
        sum_cos += np.cos(p * angle)
        sum_sin += np.sin(p * angle)

    r_x = sum_cos / len(angles)
    r_y = sum_sin / len(angles)

    R = np.sqrt(r_x**2 + r_y**2)
    alpha_m = np.arctan2(r_y, r_x) / p

    if unit == "degrees":
        cue_direction = np.deg2rad(cue_direction)

    if stats_mode == "axial":
        if cue_direction >= np.pi:
            cue_direction -= np.pi
    c = np.cos(p * (alpha_m - cue_direction))
    alpha_m = np.rad2deg(alpha_m)

    if alpha_m < 0.0:
        alpha_m += 360.0 / p

    return [alpha_m, R, c]


def compute_polarity_index_per_image(feature_df, feature_name):
    """Compute the polarity index for each image in the feature_df.

    Args:
        feature_df:
            pandas data frame with circular features
        feature_name
            str, name of circular feature
    """
    cols = ["filename", "alpha_m", "R", "c", "V"]

    polarity_index_df = pd.DataFrame(columns=cols)

    counter = 0
    for filename in feature_df["filename"].unique():
        single_image_properties_df = feature_df[feature_df["filename"] == filename]

        angles = np.array(single_image_properties_df[feature_name])
        alpha_m, R, c = compute_polarity_index(
            angles, cue_direction=0.0, stats_mode="directional", unit="radians"
        )
        polarity_index_df.at[counter, "filename"] = filename
        polarity_index_df.at[counter, "alpha_m"] = alpha_m
        polarity_index_df.at[counter, "R"] = R
        polarity_index_df.at[counter, "c"] = c
        polarity_index_df.at[counter, "V"] = c * R
        counter += 1

    return polarity_index_df
