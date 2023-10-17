"""Module for support computing statistics."""


import numpy as np

def compute_polarity_index(angles: np.ndarray, cue_direction:  float = 0.0, stats_mode: str = 'directional'):
    """Compute the polarity index of a list of angles.

    Args:
        angles:
            array with directional or axial data
        cue_direction:
            direction of the imposed cue, used for V-score computation
        stats_mode:
            'directional' or 'axial'

    Returns:
        list of values [alpha_m, R, c], alpha_m is the mean direction, R is the resultant vector length, c is a measure
        of deviation from the mean angle, used to calculate the V-score: V=cR

    """

    get_logger().info("Compute circular statistics")

    sum_cos = 0.0
    sum_sin = 0.0

    if cue_direction > 360.0:
        get_logger().warn("Warning invalid cue direction, must be between 0 and 360 degrees")

    p = 1.0
    if stats_mode == 'axial':
        p = 2.0

    for angle in angles:
        sum_cos += np.cos(p*angle)
        sum_sin += np.sin(p*angle)

    r_x = sum_cos / len(angles)
    r_y = sum_sin / len(angles)

    R = np.sqrt(r_x**2 + r_y**2)
    alpha_m = np.arctan2(r_y, r_x)/p
    cue_direction_ = np.deg2rad(cue_direction)
    if stats_mode == 'axial':
        if cue_direction_ >= np.pi:
            cue_direction_ -= np.pi
    c = np.cos(alpha_m - cue_direction_)

    alpha_m = np.rad2deg(alpha_m)

    if alpha_m < 0.0:
        alpha_m += 360.0/p

    return [alpha_m, R, c]

