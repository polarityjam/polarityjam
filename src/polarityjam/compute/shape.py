"""Collection of functions involving cell shape operations."""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from shapely.affinity import rotate
from shapely.geometry import LineString
from shapely.geometry.polygon import Polygon
from shapely.ops import split

from polarityjam.compute.compute import compute_ref_x_abs_angle_deg
from polarityjam.compute.corner import get_contour
from polarityjam.model.masks import BioMedicalMask

if TYPE_CHECKING:
    from polarityjam.model.image import BioMedicalChannel


def mask_from_contours(
    ref_img: np.ndarray, coord_list_x: np.ndarray, coord_list_y: np.ndarray
) -> np.ndarray:
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

    coord_list = []
    for a, b in zip(coord_list_x, coord_list_y):
        coord_list.append([a, b])

    mask = cv2.drawContours(
        mask, [np.array(coord_list).astype(np.int32)], -1, 1, thickness=cv2.FILLED
    )

    return mask


def partition_single_cell_mask(
    sc_mask: Union[np.ndarray, BioMedicalMask],
    cue_direction: int,
    major_axes_length: Union[int, float],
    num_partitions: int,
    contours: Optional[np.ndarray] = None,
) -> Tuple[List[np.ndarray], List[Polygon], np.ndarray]:
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
        contours:
            The contours of the single cell. If not provided, they will be computed.

    Returns:
        The list of partitioned masks counter clock wise from the cue direction
        sorted polygons counter clock wise from the cue direction
        contour of the single cell

    """
    if isinstance(sc_mask, BioMedicalMask):
        sc_mask = sc_mask.data

    # cv2 needs flipped y-axis
    sc_mask_f = np.flip(sc_mask, axis=0)

    # get the contour of the single cell mask
    if contours is None:
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
        key=lambda x: compute_ref_x_abs_angle_deg(
            pg_cent_a, pg_cent_b, x.centroid.coords.xy[0][0], x.centroid.coords.xy[1][0]
        )
        - (div_angle / 2)
        + cue_direction
    )

    masks = []
    # TODO: check if this is sufficient to catch all cases
    for s in polygons:
        if s.geom_type != "Polygon":
            warnings.warn(
                "Partition of the cell i not a Polygon."
                )
            continue

        c = s.exterior.coords.xy
        x = np.asarray(c[0].tolist()).astype(np.uint)
        y = np.asarray(c[1].tolist()).astype(np.uint)

        # flip back to original y axis
        convex_mask = np.flip(mask_from_contours(sc_mask_f, x, y), axis=0)

        # keep only concave hull parts of the mask
        masks.append((np.logical_and(convex_mask, sc_mask)).astype(np.uint8))

    if len(masks) != num_partitions:
        warnings.warn(
            "Number of partitions({}) does not match the number of created masks ({}). ".format(  # noqa: P101
                num_partitions, len(masks)
            )
        )

    return masks, polygons, contours


def get_divisor_lines(
    origin: List[int], cue_direction: int, div_line: LineString, num_partitions: int
):
    """Rotates a line from an origin point on the line based on a cue direction as often as specified.

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


def center_single_cell(
    img_list: Sequence[Union[BioMedicalChannel, BioMedicalMask, np.ndarray]],
    contours: Union[np.ndarray, Polygon],
) -> List[np.ndarray]:
    """Centers a sequence of images around the contours of a single cell.

    Args:
        img_list:
            The list of images to be centered.
        contours:
            The contours of the single cell. Either as a numpy array or a shapely polygon.

    Returns:
        The list of centered images.
    """
    if isinstance(contours, Polygon):
        x_min, y_min, x_max, y_max = _bounding_box(*contours.exterior.coords.xy)

        x = int(np.floor(x_min))
        y = int(np.floor(y_min))
        w = int(np.ceil(x_max - x_min))
        h = int(np.ceil(y_max - y_min))

    else:
        # get the bounding box of the single cell
        x, y, w, h = cv2.boundingRect(contours)

    cropped_img_list = []
    for img in img_list:

        if not isinstance(img, np.ndarray):
            img = img.data

        # flip y coordinate and treat as maximum
        y_max = img.shape[0] - y

        if y_max > img.shape[0]:
            y_max = img.shape[0]

        if x > img.shape[1]:
            x = img.shape[1]

        # crop the image to the bounding box
        img_cropped = img[y_max - h : y_max, x : x + w]  # noqa: E203

        cropped_img_list.append(img_cropped)

    return cropped_img_list


def _bounding_box(x_coordinates, y_coordinates):
    return [
        min(x_coordinates),
        min(y_coordinates),
        max(x_coordinates),
        max(y_coordinates),
    ]
