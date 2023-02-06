from typing import Union, List, Tuple

import cv2
import numpy as np
from shapely.affinity import rotate
from shapely.geometry import LineString
from shapely.geometry.polygon import Polygon
from shapely.ops import split
from collections import deque
from polarityjam.compute.compute import compute_angle_deg, compute_ref_x_abs_angle_deg
from polarityjam.compute.corner import get_contour


def mask_from_contours(ref_img: np.ndarray, coord_list_x: np.ndarray, coord_list_y: np.ndarray) -> np.ndarray:
    """Create a mask from a list of coordinates.

    Args:
        ref_img:
            The reference image
        coord_list_x:
            The x coordinates
        coord_list_y:
            The y coordinates

    Returns:
        The mask

    """
    mask = np.zeros(ref_img.shape, dtype=np.uint8)

    l = []
    for a, b in zip(coord_list_x, coord_list_y):
        l.append([a, b])

    mask = cv2.drawContours(mask, [np.array(l)], -1, (1), thickness=cv2.FILLED)

    return mask


def partition_single_cell_mask(sc_mask: np.ndarray, cue_direction: int,
                               major_axes_length: Union[int, float], num_partitions: int) -> Tuple[
    List[np.ndarray], List[Polygon]]:
    """Partitions a single cell mask into multiple masks from its centroid.

    Args:
        sc_mask:
            The single cell mask
        cue_direction:
            The orientation of the cue (e.g. flow)
        major_axes_length:
            The major axes length of the single cell
        num_partitions:
            The number of desired partitions

    Returns:
        The list of partitioned masks counter clock wise from the cue direction

    """
    # cv2 needs flipped y axis
    sc_mask = np.flip(sc_mask, axis=0)

    # get the contour of the single cell mask
    contours = get_contour(sc_mask.astype(int))
    pg = Polygon(contours)

    pg_cent_a = int(pg.centroid.coords.xy[0][0])
    pg_cent_b = int(pg.centroid.coords.xy[1][0])

    # divisor line
    a = [pg_cent_a, pg_cent_b]
    b = [pg_cent_a + int(major_axes_length), pg_cent_b]  # lies horizontally
    div_line = LineString([a, b])

    # determine number of divisions
    div_angle = int(360 / num_partitions)

    cumulative_angle = cue_direction + int(div_angle / 2)
    divisors = []
    while cumulative_angle < 360:
        divisors.append(rotate(div_line, cumulative_angle, origin=a))
        cumulative_angle += div_angle

    div_coords = []
    for div in divisors:
        div_coords += [(int(np.rint(x)), int(np.rint(y))) for x, y in [*div.coords]]

    splits = LineString(div_coords)
    sectors = split(pg, splits)

    polygons = list(sectors.geoms)

    polygons.sort(
        key=lambda x:
            compute_ref_x_abs_angle_deg(
                pg_cent_a, pg_cent_b,
                x.centroid.coords.xy[0][0], x.centroid.coords.xy[1][0]
            ) - div_angle / 2
    )

    masks = []
    for s in polygons:
        c = s.exterior.coords.xy
        x = np.asarray(c[0].tolist()).astype(np.uint)
        y = np.asarray(c[1].tolist()).astype(np.uint)

        # flip back to original y axis
        masks.append(np.flip(mask_from_contours(sc_mask, x, y), axis=0))

    return masks, polygons
