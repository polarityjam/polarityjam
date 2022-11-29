import collections
import math

import numpy as np
import skimage.filters
import skimage.measure


def compute_reference_target_orientation_rad(ref_x, ref_y, target_x, target_y):
    """Computes the marker polarity radius"""
    vec_x = ref_x - target_x
    vec_y = ref_y - target_y
    organelle_orientation_rad = np.pi - np.arctan2(vec_x, vec_y)

    return organelle_orientation_rad


def compute_angle_deg(angle_rad):
    """Returns degree"""
    return 180.0 * angle_rad / np.pi


def compute_shape_orientation_rad(orientation):
    """Computes the shape orientation with zero based on x-axis"""
    # note, the values of orientation from props are in [-pi/2,pi/2] with zero along the y-axis
    return np.pi / 2.0 + orientation


def compute_marker_vector_norm(cell_x, cell_y, marker_centroid_x, marker_centroid_y):
    """Computes the marker vector norm"""
    distance2 = (cell_x - marker_centroid_x) ** 2
    distance2 += (cell_y - marker_centroid_y) ** 2

    return np.sqrt(distance2)


def map_single_cell_to_circle(sc_protein_area, x_centroid, y_centroid, r):
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
        if (new_x in range(0,circular_img.shape[0]))  and (new_y in range(0,circular_img.shape[1])):
            circular_img[int(new_x), int(new_y)] += sc_protein_area[x, y]
            if (int(new_x), int(new_y)) not in circular_img_count.keys():
                circular_img_count[int(new_x), int(new_y)] = 1
            else:
                circular_img_count[int(new_x), int(new_y)] += 1
        #else:
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


def otsu_thresh_mask(mask, channel):
    # otsu threshold a mask given a channel from the input image
    masked_channel = mask * channel
    otsu_val = skimage.filters.threshold_otsu(masked_channel)
    masked_channel[masked_channel <= otsu_val] = 0

    return masked_channel


def compute_single_cell_prop(single_cell_mask, intensity=None):
    """Gets the single cell properties."""
    # we are looking at a single cell. There is only one region!
    regions = skimage.measure.regionprops(single_cell_mask, intensity_image=intensity)
    if len(regions) > 1:
        raise ValueError("Too many regions for a single cell!")
    props = regions[-1]

    return props


def straight_line_length(corners):
    """Computes length between corners. Corner point assumed to be in the correct order."""
    dist = []
    for idx, c in enumerate(corners):
        x = c
        y = corners[idx + 1] if idx < (len(corners) - 1) else corners[0]

        # dist between the corner to the next
        dist.append(math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2))

    return sum(dist)
