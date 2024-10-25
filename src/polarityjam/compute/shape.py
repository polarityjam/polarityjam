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
from sklearn.neighbors import KernelDensity

from polarityjam.compute.compute import (
    compute_marker_vector_norm,
    compute_ref_x_abs_angle_deg,
    compute_reference_target_orientation_rad,
)
from polarityjam.compute.corner import get_contour
from polarityjam.model.masks import BioMedicalMask
from polarityjam.polarityjam_logging import get_logger
from polarityjam.utils.decorators import experimental

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


def prepare_mirroring(img_data: np.ndarray) -> np.ndarray:
    """Pad the image with zeros to be quadratic."""
    # pad image to be quadratic
    max_dim = max(img_data.shape) * 2
    pad_d1 = (max_dim - img_data.shape[0]) // 2
    pad_d2 = (max_dim - img_data.shape[1]) // 2

    pad = ((pad_d1, pad_d1), (pad_d2, pad_d2), (0, 0))

    return np.pad(
        img_data,
        pad,  # pad evenly on both sides
        mode="constant",
        constant_values=0,
    )


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

    # expand the mask to 3D if it is 2D
    if img_data.ndim == 2:
        img_data = img_data[..., None]

    # pad image to ensure mirrored image does not leave the frame
    img_data = prepare_mirroring(img_data)

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

    # conditional sign flip
    # flip_sign_matrix = np.ones(c1.shape)
    # flip_sign_matrix[c1 < 0] = 1

    reflect_coordinates_b2 = (
        b2.reshape(2, 1) * c2
    )  # calculate the points on the second axis conditionally flipped

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


def normalized_center_distance(origin: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Calculate the normalized distance of each point to the origin."""
    _p = [
        compute_marker_vector_norm(
            origin[0],
            origin[1],
            p[0],
            p[1],
        )
        for p in points
    ]
    return np.array(_p) / np.max(_p)


def get_shanon_estimated_pdf_center_distance(
    origin: np.ndarray,
    points: np.ndarray,
    bins: Union[int, str] = "auto",
    plot: bool = False,
) -> float:
    """Calculate the distance probability density function of each point to the origin."""
    n_center_dist = normalized_center_distance(origin, points)
    h, bin_edges = np.histogram(n_center_dist, bins=bins, range=(0, 1))

    # plot the histogram
    if plot:
        import matplotlib.pyplot as plt

        plt.hist(n_center_dist, bins=bins, range=(0, 1))
        plt.show()

    # use kernel density estimation to get the pdf
    kd_data1 = KernelDensity(kernel="gaussian", bandwidth=0.05).fit(
        n_center_dist.reshape((-1, 1))
    )

    resolution = 14

    x = np.linspace(min(n_center_dist), max(n_center_dist), np.power(2, resolution))[
        :, np.newaxis
    ]

    # get the estimated density
    kd_vals_data1 = np.exp(kd_data1.score_samples(x)) / np.power(2, resolution)

    # plot the estimated density
    if plot:
        plt.plot(x, kd_vals_data1, markersize=4)
        plt.show()

    # calculate the shannon entropy
    shannon_entropy = -np.sum(kd_vals_data1 * np.log2(kd_vals_data1))

    return shannon_entropy


@experimental
def get_shanon_estimated_pdf_angle(
    points: np.ndarray, bins: Union[int, str] = "auto", plot: bool = False
) -> float:
    """Calculate the angle probability density function of each point to the origin."""
    angles = []
    for i in points:
        _neighbors = [
            points[j] for j in range(len(points)) if not np.array_equal(points[j], i)
        ]

        # get the nearest neighbor of point i
        nn_1 = min(
            _neighbors, key=lambda x: compute_marker_vector_norm(i[0], i[1], x[0], x[1])
        )

        # remove the nearest neighbor from the ndarray
        _neighbors = [
            _neighbors[j]
            for j in range(len(_neighbors))
            if not np.array_equal(_neighbors[j], nn_1)
        ]

        # compute reference point
        p0 = i[0] + (i[0] - nn_1[0])
        p1 = i[1] + (i[1] - nn_1[1])

        # search for the nearest point to the prediction point
        nn_2 = min(
            _neighbors, key=lambda x: compute_marker_vector_norm(p0, p1, x[0], x[1])
        )

        # point i serves as the origin
        nn_1_o = [nn_1[0] - i[0], nn_1[1] - i[1]]
        nn_2_o = [nn_2[0] - i[0], nn_2[1] - i[1]]

        _angle = compute_reference_target_orientation_rad(
            nn_1_o[0], nn_2_o[0], nn_1_o[1], nn_2_o[1]
        )

        angle = _angle if _angle <= np.pi else 2 * np.pi - _angle

        # append the angle to the list
        angles.append(angle)

    if plot:
        import matplotlib.pyplot as plt

        plt.hist(angles, bins=bins)
        plt.show()

    # use kernel density estimation to get the pdf
    kd_data1 = KernelDensity(kernel="gaussian", bandwidth=0.25).fit(
        np.array(angles).reshape((-1, 1))
    )

    resolution = 18

    x = np.linspace(0, np.pi, np.power(2, resolution))[:, np.newaxis]

    # get the estimated density
    kd_vals_data1 = np.exp(kd_data1.score_samples(x)) / np.power(2, resolution)

    # plot the estimated density
    if plot:
        plt.plot(x, kd_vals_data1, markersize=4)
        plt.show()

    # calculate the shannon entropy
    shannon_entropy = -np.sum(kd_vals_data1 * np.log2(kd_vals_data1))

    return shannon_entropy
