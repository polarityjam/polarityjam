"""Collection of functions involving cell shape operations."""
from __future__ import annotations

import itertools
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
from polarityjam.polarityjam_logging import get_logger

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


def _correct_for_cue_direction(angle, cue_direction):
    """Corrects an angle for a given cue direction. This is necessary to sort the partitions."""
    if angle - cue_direction < 0:
        return angle - cue_direction + 360
    else:
        return angle - cue_direction


def _correct_for_num_slices(angle, div_angle):
    """Corrects an angle for a given number of slices.

    Corrects such that sorting is possible based on the
    center points of the partition relative to the middle point of the cell.
    """
    if angle + int(div_angle / 2) > 360:  # must be floored
        return angle + int(div_angle / 2) - 360
    else:
        return angle + int(div_angle / 2)


def partition_single_cell_mask(
    sc_mask: Union[np.ndarray, BioMedicalMask],
    cue_direction: int,
    major_axes_length: Union[int, float],
    num_partitions: int,
    contours: Optional[np.ndarray] = None,
) -> Tuple[List[np.ndarray], List[Polygon], np.ndarray]:
    """Partitions a single cell mask into multiple masks from its centroid. Number of partitions.

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
        The list of partitioned masks sorted counter clock wise from the cue direction
        counter clock wise in cue direction sorted polygons
        contour of the single cell

    """
    if num_partitions > 359:
        raise ValueError("Number of partitions must be smaller than 360.")

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

    # turn this line based on the cue direction
    div_line = rotate(div_line, cue_direction, origin=(pg_cent_a, pg_cent_b))

    # determine number of divisions
    divisors, div_angle = get_divisor_lines(a, div_line, num_partitions)

    div_coords = []
    for div in divisors:
        div_coords += [(int(np.rint(x)), int(np.rint(y))) for x, y in [*div.coords]]

    splits = LineString(div_coords)
    sectors = split(pg_convex, splits)

    polygons = list(sectors.geoms)

    # sort polygons based on their angle to the cell center and polygon center,
    # correct for cue direction and number of partitions before sorting
    polygons.sort(
        key=lambda x: _correct_for_num_slices(
            _correct_for_cue_direction(
                compute_ref_x_abs_angle_deg(
                    pg_cent_a,
                    pg_cent_b,
                    x.centroid.coords.xy[0][0],
                    x.centroid.coords.xy[1][0],
                ),
                cue_direction,
            ),
            div_angle,
        )
    )

    masks = []
    # TODO: check if this is sufficient to catch all cases
    for s in polygons:
        if s.geom_type != "Polygon":
            get_logger().warn("Partition of the cell is not a Polygon.")
            continue

        c = s.exterior.coords.xy
        x = np.asarray(c[0].tolist()).astype(np.uint)
        y = np.asarray(c[1].tolist()).astype(np.uint)

        # flip back to original y axis
        convex_mask = np.flip(mask_from_contours(sc_mask_f, x, y), axis=0)

        # keep only concave hull parts of the mask
        masks.append((np.logical_and(convex_mask, sc_mask)).astype(np.uint8))

    if len(masks) != num_partitions:
        get_logger().warn(
            "Number of partitions({}) does not match the number of created masks ({}). ".format(  # noqa: P101
                num_partitions, len(masks)
            )
        )

    return masks, polygons, contours


def get_divisor_lines(origin: List[int], div_line: LineString, num_partitions: int):
    """Rotates a line from an origin point on the line based on a cue direction as often as specified.

    Args:
        origin:
            origin point on the line
        div_line:
            shapely line object
        num_partitions:
            The number of desired partitions

    Returns:
        The list of partitioned masks counter clock wise from the cue direction

    """
    div_angle = int(360 / num_partitions)
    cumulative_angle = int(div_angle / 2)
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


def mirror_along_cue_direction(
    img: Union[np.ndarray, BioMedicalMask],
    mirror_angle_in_deg: int,
    origin: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Mirror the given image along the cue direction."""
    if isinstance(img, BioMedicalMask):
        img_data = img.data
    else:
        img_data = img

    # pad image to be quadratic
    if img_data.shape[0] != img_data.shape[1]:
        max_dim = max(img_data.shape) + 2  # always pad 2 to avoid rounding issues
        pad_d1 = (max_dim - img_data.shape[0]) // 2
        pad_d2 = (max_dim - img_data.shape[1]) // 2
        img_data = np.pad(
            img_data,
            ((pad_d1, pad_d1), (pad_d2, pad_d2)),  # pad evenly on both sides
            mode="constant",
            constant_values=0,
        )

    # expand the mask to 3D if it is 2D
    if img_data.ndim == 2:
        img_data = img_data[..., None]

    if origin is None:
        contours = get_contour(img_data.astype(int))

        pg = Polygon(contours)  # swaps x and y

        pg_cent_a = int(pg.centroid.coords.xy[0][0])
        pg_cent_b = int(pg.centroid.coords.xy[1][0])

        origin = np.array([pg_cent_b, pg_cent_a], dtype="float")

    unit_cue_x = np.cos(np.deg2rad(mirror_angle_in_deg))
    unit_cue_y = np.sin(np.deg2rad(mirror_angle_in_deg))

    b1 = np.array(
        [unit_cue_x, unit_cue_y], dtype="float"
    )  # unit vector of the cue direction
    b2 = np.array(
        [-unit_cue_y, unit_cue_x], dtype="float"
    )  # unit vector orthogonal to the cue direction

    points = np.moveaxis(np.indices(img_data.shape[:2]), 0, -1).reshape(
        -1, 2
    )  # get all points of the image as indices

    list_origin = origin.reshape(1, 2)
    c1 = (
        points - list_origin
    ) @ b1  # get the coefficient of the points in their new basis b1
    c2 = (
        points - list_origin
    ) @ b2  # get the coefficient of the points in their new basis b2

    reflect_coordinates_b1 = b1.reshape(2, 1) * np.abs(
        c1
    )  # calculate reflection of the points on the first axis
    reflect_coordinates_b2 = (
        b2.reshape(2, 1) * c2
    )  # calculate the points on the second axis

    # get the coordinates of the reflected points in the new basis by adding back the origin
    reflection_coordinates = (
        reflect_coordinates_b1 + reflect_coordinates_b2 + origin.reshape(2, 1)
    )

    # build fractional part of the new coordinates
    frac_x = reflection_coordinates[0, :] % 1  # fractional part of the new x
    frac_x_ = 1 - frac_x  # 1 - fractional part of the new x
    frac_y = reflection_coordinates[1, :] % 1  # fractional part of the new y
    frac_y_ = 1 - frac_y  # 1 - fractional part of the new y

    weights_x = [
        (frac_x_).reshape(*reflection_coordinates[0, :].shape, 1),
        (frac_x).reshape(*reflection_coordinates[0, :].shape, 1),
    ]  # weights for the x-axis are 1 - fractional part and fractional part of the new x
    weights_y = [
        (frac_y_).reshape(*reflection_coordinates[1, :].shape, 1),
        (frac_y).reshape(*reflection_coordinates[1, :].shape, 1),
    ]  # weights for the y-axis are 1 - fractional part and fractional part of the new y

    start_x = np.floor(reflection_coordinates[0, :])  # start x index
    start_y = np.floor(reflection_coordinates[1, :])  # start y index

    max_index_x = img_data.shape[0] - 1
    max_index_y = img_data.shape[1] - 1

    # new image is the sum of the weighted values of the 4 points around the new point
    transformed_image = np.sum(
        [
            img_data[
                np.clip(np.floor(start_x + x), 0, max_index_x).astype("int"),
                np.clip(np.floor(start_y + y), 0, max_index_y).astype("int"),
            ]
            * weights_x[x]
            * weights_y[y]
            for x, y in itertools.product([0, 1], repeat=2)  # cartesian product
        ],
        axis=0,
    )
    transformed_image = transformed_image.reshape(*img_data.shape)

    return transformed_image.squeeze()
