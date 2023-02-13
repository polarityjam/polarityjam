from typing import Union, List, Tuple

import cv2
import numpy as np
from shapely.affinity import rotate
from shapely.geometry import LineString
from shapely.geometry.polygon import Polygon
from shapely.ops import split

from polarityjam.compute.compute import compute_ref_x_abs_angle_deg
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

    mask = cv2.drawContours(mask, [np.array(l).astype(np.int32)], -1, 1, thickness=cv2.FILLED)

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
    # cv2 needs flipped y-axis
    sc_mask_f = np.flip(sc_mask, axis=0)

    # get the contour of the single cell mask
    contours = get_contour(sc_mask_f.astype(int))
    pg = Polygon(contours)
    pg_convex = pg.convex_hull

    pg_cent_a = int(pg.centroid.coords.xy[0][0])
    pg_cent_b = int(pg.centroid.coords.xy[1][0])

    # divisor line
    a = [pg_cent_a, pg_cent_b]
    b = [pg_cent_a + int(major_axes_length), pg_cent_b]  # lies horizontally
    div_line = LineString([a, b])

    # determine number of divisions
    divisors, div_angle = get_divisor_lines(a, cue_direction, div_line, num_partitions)

    div_coords = []
    for div in divisors:
        div_coords += [(int(np.rint(x)), int(np.rint(y))) for x, y in [*div.coords]]

    splits = LineString(div_coords)
    sectors = split(pg_convex, splits)

    polygons = list(sectors.geoms)

    # sort polygons counter clock wise
    polygons.sort(
        key=lambda x:
        compute_ref_x_abs_angle_deg(
            pg_cent_a, pg_cent_b,
            x.centroid.coords.xy[0][0], x.centroid.coords.xy[1][0]
        ) - (div_angle / 2) + cue_direction
    )

    masks = []
    for s in polygons:
        c = s.exterior.coords.xy
        x = np.asarray(c[0].tolist()).astype(np.uint)
        y = np.asarray(c[1].tolist()).astype(np.uint)

        # flip back to original y axis
        convex_mask = np.flip(mask_from_contours(sc_mask_f, x, y), axis=0)

        # keep only concave hull parts of the mask
        masks.append((np.logical_and(convex_mask, sc_mask)).astype(np.uint8))

    assert len(masks) == num_partitions, "Number of partitions(%s) does not match the number of masks (%s)." % (
        num_partitions, len(masks))

    return masks, polygons


def get_divisor_lines(origin, cue_direction, div_line, num_partitions):
    """Rotates a line from an origin point on the line based on a cue direction as often as specified

    Args:
        origin:
            origin point on the line
        cue_direction:
            The orientation of the cue (e.g. flow)
        div_line:
            shapely line object
        num_partitions:
            The number of desired partitions

    Returns:
        The list of partitioned masks counter clock wise from the cue direction

    """
    div_angle = int(360 / num_partitions)
    cumulative_angle = cue_direction + int(div_angle / 2)
    divisors = []
    while cumulative_angle < 360:
        divisors.append(rotate(div_line, cumulative_angle, origin=origin))
        cumulative_angle += div_angle
    return divisors, div_angle
