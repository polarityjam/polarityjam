"""Module for support computing statistics."""


import numpy as np
from polarityjam.polarityjam_logging import get_logger

def compute_polarity_index(angles: np.ndarray, cue_direction:  float = 0.0, stats_mode: str = 'directional', unit: str = 'radians'):
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
        list of values [alpha_m, R, c], alpha_m is the mean direction (in degrees), R is the resultant vector length, c is a measure
        of deviation from the mean angle, used to calculate the V-score: V=cR

    """

    get_logger().info("Compute circular statistics")

    sum_cos = 0.0
    sum_sin = 0.0

    #if cue_direction > 360.0:
    #    get_logger().warn("Warning invalid cue direction, must be between 0 and 360 degrees")

    p = 1.0
    if stats_mode == 'axial':
        p = 2.0

    if unit == 'degrees':
        angles = np.deg2rad(angles)

    for angle in angles:
        sum_cos += np.cos(p*angle)
        sum_sin += np.sin(p*angle)

    r_x = sum_cos / len(angles)
    r_y = sum_sin / len(angles)

    R = np.sqrt(r_x**2 + r_y**2)
    alpha_m = np.arctan2(r_y, r_x)/p

    if unit == 'degrees':
        cue_direction = np.deg2rad(cue_direction)

    if stats_mode == 'axial':
        if cue_direction >= np.pi:
            cue_direction -= np.pi
    c = np.cos(alpha_m - cue_direction)

    alpha_m = np.rad2deg(alpha_m)

    if alpha_m < 0.0:
        alpha_m += 360.0/p

    return [alpha_m, R, c]

