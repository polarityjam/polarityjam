import math
from typing import List, Tuple

import numpy as np

from polarityjam.utils.decorators import experimental


def compute_reference_target_orientation_rad(ref_x: float, ref_y: float, target_x: float, target_y: float) -> float:
    """Computes the 2D orientation in rad between reference and target.

    Args:
        ref_x:
            The x coordinate of the reference point
        ref_y:
            The y coordinate of the reference point
        target_x:
            The x coordinate of the target point
        target_y:
            The y coordinate of the target point

    Returns:
        The orientation in rad between reference and target

    """
    vec_x = ref_x - target_x
    vec_y = ref_y - target_y
    organelle_orientation_rad = np.pi - np.arctan2(vec_x, vec_y)

    return organelle_orientation_rad


def compute_ref_x_abs_angle_deg(ref_x: float, ref_y: float, x: float, y: float) -> float:
    """Computes the angle between the x-axis and the vector (x,y).

    Args:
        ref_y:
            The y coordinate of the reference point
        ref_x:
            The x coordinate of the reference point
        x:
            The x coordinate
        y:
            The y coordinate

    Returns:
        The angle in rad

    """
    y_vec = y - ref_y
    x_vec = x - ref_x

    angle_deg = compute_angle_deg(np.arctan2(y_vec, x_vec))

    if angle_deg < 0:
        angle_deg = (180 - abs(angle_deg)) + 180

    return angle_deg


def compute_angle_deg(angle_rad: float) -> float:
    """Computes the angle given in rad in degrees.

    Args:
        angle_rad:
            The angle in rad

    Returns:
        The angle in degrees

    """
    return 180.0 * angle_rad / np.pi


def compute_shape_orientation_rad(orientation: float) -> float:
    """Computes the shape orientation (zero based) on x-axis.

    Args:
        orientation:
            The orientation

    Returns:
        The shape orientation in rad

    """
    # note, the values of orientation from props are in [-pi/2,pi/2] with zero along the y-axis
    return np.pi / 2.0 + orientation


def compute_marker_vector_norm(cell_x: float, cell_y: float, marker_centroid_x: float,
                               marker_centroid_y: float) -> float:
    """Computes the marker vector norm.

    Args:
        cell_x:
            The x coordinate of the cell
        cell_y:
            The y coordinate of the cell
        marker_centroid_x:
            The x coordinate of the marker centroid
        marker_centroid_y:
            The y coordinate of the marker centroid

    Returns:
        The marker vector norm

    """
    distance2 = (cell_x - marker_centroid_x) ** 2
    distance2 += (cell_y - marker_centroid_y) ** 2

    return np.sqrt(distance2)


@experimental
def map_single_cell_to_circle(sc_protein_area, x_centroid, y_centroid, r):
    """Maps a single cell to a circle. NOTE: EXPERIMENTAL"""

    circular_img = np.zeros([sc_protein_area.shape[0], sc_protein_area.shape[1]])
    circular_img_count = {}

    x_nonzero, y_nonzero = np.nonzero(sc_protein_area)

    # loop over bounding box indices
    for x, y in zip(x_nonzero, y_nonzero):
        x_vec = x_centroid - x
        y_vec = y_centroid - y

        angle_rad = np.pi - np.arctan2(x_vec, y_vec)
        new_x = r * np.sin(angle_rad) + x_centroid
        new_y = r * np.cos(angle_rad) + y_centroid

        # correct for wrong x axis alignment (bottom left corner is (0,0), not top left)
        new_x = x_centroid - (new_x - x_centroid)
        new_x = int(new_x)
        new_y = int(new_y)

        # count, TODO: check why new_x and new_y are sometimes out of the image boundaries
        if (new_x in range(0, circular_img.shape[0])) and (new_y in range(0, circular_img.shape[1])):
            circular_img[int(new_x), int(new_y)] += sc_protein_area[x, y]
            if (int(new_x), int(new_y)) not in circular_img_count.keys():
                circular_img_count[int(new_x), int(new_y)] = 1
            else:
                circular_img_count[int(new_x), int(new_y)] += 1
        # else:
        #    print("Circular image coords out of bounds")
        #    print("x: ", int(new_x), ", y:", int(new_y))
    #        if (int(new_x), int(new_y)) not in circular_img_count.keys():
    #            circular_img_count[int(new_x), int(new_y)] = 1
    #        else:
    #            circular_img_count[int(new_x), int(new_y)] += 1
    #
    #        circular_img[int(new_x), int(new_y)] += sc_protein_area[x, y]

    # calculate mean
    for k in circular_img_count.keys():
        circular_img[k[0], k[1]] = circular_img[k[0], k[1]] / circular_img_count[k]

    return circular_img


def straight_line_length(corners: List[Tuple[int, int]]) -> float:
    """Computes length between corners. Corner point assumed to be in the correct order.

    Args:
        corners:
            The corners of the cell in the correct order

    Returns:
        The straight line length between the consecutive corner points
    """
    dist = []
    for idx, c in enumerate(corners):
        x = c
        y = corners[idx + 1] if idx < (len(corners) - 1) else corners[0]

        # dist between the corner to the next
        dist.append(math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2))

    return sum(dist)
