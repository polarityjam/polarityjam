import math

import cv2
import numpy as np


def get_corner(sc_img, epsilon=5):
    contours, _ = cv2.findContours(sc_img.astype(bool).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    idx_l = 0
    len_l = 0
    if len(contours) > 1:
        # only take the longest contour
        for idx, c in enumerate(contours):
            if len(c) > len_l:
                idx_l = idx
                len_l = len(c)

    contours = contours[idx_l].squeeze(1)

    # append first point to build a circular structure
    contours = np.concatenate((contours, [contours[0]]))

    corners = douglas_peucker(contours, epsilon)

    return corners[:-1][:]  # last corner point forms circle


def douglas_peucker(points, epsilon):
    if len(points) < 3:
        return points

    dmax = 0
    index = -1
    end = len(points) - 1

    for idx in range(1, end):
        d = perpendicular_distance(points[idx], points[0], points[end])

        if d > dmax:
            index = idx
            dmax = d

    if dmax > epsilon:
        rec_result1 = douglas_peucker(points[0:index + 1], epsilon)
        rec_result2 = douglas_peucker(points[index:], epsilon)

        result = np.concatenate((rec_result1[0:len(rec_result1) - 1], rec_result2), axis=0)

        return result

    else:
        return np.array([points[0], points[end]])


def perpendicular_distance(p, p1, p2):
    if p1[0] == p2[0]:
        result = abs(p[0] - p1[0])
    else:
        slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
        intercept = p1[1] - (slope * p1[0])

        result = abs(slope * p[0] - p[1] + intercept) / math.sqrt(math.pow(slope, 2) + 1)

    return result
