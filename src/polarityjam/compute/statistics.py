import numpy as np

def compute_polarity_index(angles: np.ndarray, stats_mode: str = 'directional', cue_direction:  float = 0.0):

    sum_cos = 0.0
    sum_sin = 0.0

    for angle in angles:
        sum_cos += np.cos(angle)
        sum_sin += np.sin(angle)

    r_x = sum_cos / len(angles)
    r_y = sum_sin / len(angles)

    R = np.sqrt(r_x**2 + r_y**2)
    alpha_m = np.arctan2(r_y, r_x)
    cue_direction_ = np.deg2rad(cue_direction)
    c = np.cos(alpha_m - cue_direction_)

    alpha_m = np.rad2deg(alpha_m)

    if alpha_m < 0.0:
        alpha_m += 360.0

    return [alpha_m, R, c]

