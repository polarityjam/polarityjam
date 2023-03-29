"""Hold all classes and functions related to the properties of a cell."""
from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
import skimage.measure
from scipy import ndimage as ndi
from skimage.measure._regionprops import RegionProperties

from polarityjam.compute.compute import (
    compute_angle_deg,
    compute_marker_vector_norm,
    compute_reference_target_orientation_rad,
    compute_shape_orientation_rad,
    straight_line_length,
)
from polarityjam.compute.corner import get_contour, get_corner
from polarityjam.compute.shape import center_single_cell, partition_single_cell_mask
from polarityjam.model.image import BioMedicalChannel
from polarityjam.model.masks import BioMedicalMask
from polarityjam.model.parameter import RuntimeParameter


class SingleInstanceProps(RegionProperties):
    """Base class for all single cell properties."""

    def __init__(
        self,
        single_cell_mask: Union[np.ndarray, BioMedicalMask],
        intensity: Optional[Union[np.ndarray, BioMedicalChannel]] = None,
    ):
        """Initialize the properties with the given mask and intensity."""
        if isinstance(single_cell_mask, BioMedicalMask):
            single_cell_mask = single_cell_mask.to_instance_mask(1).data
        else:
            if not np.issubdtype(single_cell_mask.dtype, np.integer):
                raise RuntimeError("Only integer images allowed!")

        if isinstance(intensity, BioMedicalChannel):
            intensity = intensity.data

        objects = ndi.find_objects(single_cell_mask)

        assert len(objects) == 1, "Only one object allowed!"

        sl = objects[0]

        self.mask = single_cell_mask
        self.intensity = intensity

        super().__init__(sl, 1, single_cell_mask, intensity, True)


class SingleCellProps(SingleInstanceProps):
    """Class representing the properties of a single cell."""

    def __init__(self, single_cell_mask: BioMedicalMask, param: RuntimeParameter):
        """Initialize the properties with the given mask and intensity."""
        self.param = param
        super().__init__(single_cell_mask)

    @property
    def cell_shape_orientation_rad(self):
        """Return the cell shape orientation in radians."""
        # note, the values of orientation from props are in [-pi/2,pi/2] with zero along the y-axis
        return compute_shape_orientation_rad(self.orientation)

    @property
    def cell_shape_orientation_deg(self):
        """Return the cell shape orientation in degrees."""
        # note, the values of orientation from props are in [-pi/2,pi/2] with zero along the y-axis
        return compute_angle_deg(self.cell_shape_orientation_rad)

    @property
    def cell_major_to_minor_ratio(self):
        """Return the major to minor ratio of the cell."""
        return self.major_axis_length / self.minor_axis_length

    @property
    def cell_corner_points(self):
        """Return the corner points of the cell."""
        return get_corner(self.mask, self.param.dp_epsilon)


class SingleCellNucleusProps(SingleInstanceProps):
    """Class representing the properties of a single nucleus."""

    def __init__(self, single_nucleus_mask: BioMedicalMask, sc_props: SingleCellProps):
        """Initialize the properties with the given mask and intensity."""
        super().__init__(single_nucleus_mask)

        self._sc_props = sc_props

    @property
    def nuc_displacement_orientation_rad(self):
        """Return the displacement orientation of the nucleus in radians."""
        return compute_reference_target_orientation_rad(
            self._sc_props.centroid[0],
            self._sc_props.centroid[1],
            self.centroid[0],
            self.centroid[1],
        )

    @property
    def nuc_displacement_orientation_deg(self):
        """Return the displacement orientation of the nucleus in degrees."""
        return compute_angle_deg(self.nuc_displacement_orientation_rad)

    @property
    def nuc_shape_orientation_rad(self):
        """Return the nucleus shape orientation in radians."""
        # note, the values of orientation from props are in [-pi/2,pi/2] with zero along the y-axis
        return compute_shape_orientation_rad(self.orientation)

    @property
    def nuc_shape_orientation_deg(self):
        """Return the nucleus shape orientation in degrees."""
        return compute_angle_deg(self.nuc_shape_orientation_rad)

    @property
    def nuc_major_to_minor_ratio(self):
        """Return the major to minor ratio of the nucleus."""
        return self.major_axis_length / self.minor_axis_length

    @property
    def contour_points(self):
        """Return the contour points of the nucleus."""
        return get_contour(self.mask)


class SingleCellOrganelleProps(SingleInstanceProps):
    """Class representing the properties of a single organelle."""

    def __init__(
        self,
        single_organelle_mask: BioMedicalMask,
        nucleus_props: SingleCellNucleusProps,
    ):
        """Initialize the properties with the given mask and intensity."""
        super().__init__(single_organelle_mask)

        self._nucleus_props = nucleus_props

    @property
    def nuc_organelle_distance(self):
        """Return the distance between the nucleus and the organelle."""
        return compute_marker_vector_norm(
            self.centroid[0],
            self.centroid[1],
            self._nucleus_props.centroid[0],
            self._nucleus_props.centroid[1],
        )

    @property
    def organelle_orientation_rad(self):
        """Return the orientation of the organelle in radians."""
        return compute_reference_target_orientation_rad(
            self._nucleus_props.centroid[0],
            self._nucleus_props.centroid[1],
            self.centroid[0],
            self.centroid[1],
        )

    @property
    def organelle_orientation_deg(self):
        """Return the orientation of the organelle in degrees."""
        return compute_angle_deg(self.organelle_orientation_rad)

    @property
    def contour_points(self):
        """Return the contour points of the organelle."""
        return get_contour(self.mask)


class SingleCellMarkerProps(SingleInstanceProps):
    """Class representing the properties of a single cell marker signal."""

    def __init__(
        self,
        single_cell_mask: BioMedicalMask,
        im_marker: BioMedicalChannel,
        cue_direction: int,
    ):
        """Initialize the properties with the given mask and intensity."""
        super().__init__(single_cell_mask, im_marker)
        quadrant_masks, _partition_polygons, contours = partition_single_cell_mask(
            single_cell_mask.data, cue_direction, self.axis_major_length, 4
        )
        # quadrant_masks_centered = [center_single_cell([q], p) for q, p in zip(quadrant_masks, _partition_polygons)]
        self.quadrant_masks = center_single_cell(quadrant_masks, contours)

        half_masks, _partition_polygons, _ = partition_single_cell_mask(
            single_cell_mask.data, cue_direction, self.axis_major_length, 2
        )
        # half_masks_centered = [center_single_cell([h], p) for h, p in zip(half_masks, _partition_polygons)]
        self.half_masks = center_single_cell(
            half_masks, contours
        )  # [item for sublist in half_masks_centered for item in sublist]

        self.intensity_cropped, self.mask_cropped = center_single_cell(
            [im_marker, single_cell_mask], contours
        )

    @property
    def marker_centroid_orientation_rad(self):
        """Return the orientation of the mass of the marker centroid in radians."""
        return compute_reference_target_orientation_rad(
            self.centroid[0],
            self.centroid[1],
            self.weighted_centroid[0],
            self.weighted_centroid[1],
        )

    @property
    def marker_centroid_orientation_deg(self):
        """Return the orientation of the mass of the marker centroid in degrees."""
        return compute_angle_deg(self.marker_centroid_orientation_rad)

    @property
    def marker_sum_expression(self):
        """Return the sum of the marker expression."""
        return self.mean_intensity * self.area

    @property
    def marker_cue_directional_intensity_ratio(self):
        """Return the ratio of the left vs right cell marker intensity in cue direction."""
        sc_marker_intensity_mask = BioMedicalMask.from_threshold_otsu(
            self.intensity_cropped * self.mask_cropped,
            gaussian_filter=None,
            rolling_ball_radius=self.intensity_cropped.shape[0] // 100,
        )
        sc_marker_intensity_mask_r = sc_marker_intensity_mask.combine(
            BioMedicalMask(self.half_masks[0])
        ).mask_background()
        sc_marker_intensity_mask_l = sc_marker_intensity_mask.combine(
            BioMedicalMask(self.half_masks[1])
        ).mask_background()

        right = sc_marker_intensity_mask_r.data * self.intensity_cropped
        left = sc_marker_intensity_mask_l.data * self.intensity_cropped

        left_m = 0 if np.ma.count_masked(left) == left.size else np.mean(left)
        right_m = 0 if np.ma.count_masked(right) == right.size else np.mean(right)

        if left_m == 0 and right_m == 0:
            warnings.warn("Warning: entire cell masked.", stacklevel=2)
            return 0

        return 1 - 2 * left_m / (left_m + right_m)

    @property
    def marker_cue_undirectional_intensity_ratio(self):
        """The ratio of the sum of cell marker quarters in cue direction and the total marker intensity."""
        sc_marker_intensity_mask = BioMedicalMask.from_threshold_otsu(
            self.intensity_cropped * self.mask_cropped,
            gaussian_filter=None,
            rolling_ball_radius=self.intensity_cropped.shape[0] // 100,
        )
        sc_marker_intensity_mask_t = sc_marker_intensity_mask.combine(
            BioMedicalMask(self.quadrant_masks[0])
        ).mask_background()
        sc_marker_intensity_mask_l = sc_marker_intensity_mask.combine(
            BioMedicalMask(self.quadrant_masks[1])
        ).mask_background()
        sc_marker_intensity_mask_b = sc_marker_intensity_mask.combine(
            BioMedicalMask(self.quadrant_masks[2])
        ).mask_background()
        sc_marker_intensity_mask_r = sc_marker_intensity_mask.combine(
            BioMedicalMask(self.quadrant_masks[3])
        ).mask_background()

        top = sc_marker_intensity_mask_t.data * self.intensity_cropped
        left = sc_marker_intensity_mask_l.data * self.intensity_cropped
        bottom = sc_marker_intensity_mask_b.data * self.intensity_cropped
        right = sc_marker_intensity_mask_r.data * self.intensity_cropped

        left_m = 0 if np.ma.count_masked(left) == left.size else np.mean(left)
        right_m = 0 if np.ma.count_masked(right) == right.size else np.mean(right)
        top_m = 0 if np.ma.count_masked(top) == top.size else np.mean(top)
        bottom_m = 0 if np.ma.count_masked(bottom) == bottom.size else np.mean(bottom)

        if sum([left_m, right_m, top_m, bottom_m]) == 0:
            warnings.warn("Warning: entire cell masked.", stacklevel=2)
            return 0

        return (top_m + bottom_m) / (left_m + right_m + top_m + bottom_m)


class SingleCellMarkerMembraneProps(SingleInstanceProps):
    """Class representing the properties of a single cell membrane signal."""

    def __init__(
        self, single_membrane_mask: BioMedicalMask, im_marker: BioMedicalChannel
    ):
        """Initialize the properties with the given mask and intensity."""
        super().__init__(single_membrane_mask, im_marker)

    @property
    def marker_sum_expression_mem(self):
        """Return the sum of the marker membrane expression."""
        return self.mean_intensity * self.area


class SingleCellMarkerNucleiProps(SingleInstanceProps):
    """Class representing the properties of a single cell marker nucleus signal."""

    def __init__(
        self,
        single_nucleus_mask: BioMedicalMask,
        im_marker: BioMedicalChannel,
        sc_nucleus_props: SingleCellNucleusProps,
        sc_marker_props: SingleCellMarkerProps,
    ):
        """Initialize the properties with the given mask and intensity."""
        super().__init__(single_nucleus_mask, im_marker)
        self._sc_nucleus_props = sc_nucleus_props
        self._sc_marker_props = sc_marker_props

    @property
    def marker_nucleus_orientation_rad(self):
        """Return the orientation of the marker nucleus in radians."""
        return compute_reference_target_orientation_rad(
            self._sc_nucleus_props.centroid[0],
            self._sc_nucleus_props.centroid[1],
            self._sc_marker_props.centroid[0],
            self._sc_marker_props.centroid[1],
        )

    @property
    def marker_nucleus_orientation_deg(self):
        """Return the orientation of the marker nucleus in degrees."""
        return compute_angle_deg(self.marker_nucleus_orientation_rad)

    @property
    def marker_sum_expression_nuc(self):
        """Return the sum of the marker nucleus expression."""
        return self.mean_intensity * self.area


class SingleCellMarkerCytosolProps(SingleInstanceProps):
    """Class representing the properties of a single cell marker cytosol signal."""

    def __init__(
        self,
        single_cytosol_mask: BioMedicalMask,
        im_marker: BioMedicalChannel,
        sc_marker_nuclei_props: SingleCellMarkerNucleiProps,
    ):
        """Initialize the properties with the given mask and intensity."""
        super().__init__(single_cytosol_mask, im_marker)
        self.sc_marker_nuclei_props = sc_marker_nuclei_props

    @property
    def marker_sum_expression_cyt(self):
        """Return the sum of the marker cytosol expression."""
        return self.mean_intensity * self.area

    @property
    def marker_cytosol_ratio(self):
        """Return the ratio of the marker cytosol expression and the marker nucleus mean intensity."""
        return self.sc_marker_nuclei_props.mean_intensity / self.mean_intensity


class SingleCellJunctionProps:
    """Class representing the properties of a single cell junction."""

    # Junction properties subclasses
    class SingleCellJunctionInterfaceProps(SingleInstanceProps):
        """Class representing the properties of a single cell junction interface."""

        # Based on junction mapper: https://doi.org/10.7554/eLife.45413
        def __init__(
            self, single_membrane_mask: BioMedicalMask, im_junction: BioMedicalChannel
        ):
            """Initialize the properties with the given mask and intensity."""
            super().__init__(single_membrane_mask, im_junction)

    class SingleCellJunctionIntensityProps(SingleInstanceProps):
        """Class representing the properties of a single cell junction intensity."""

        def __init__(
            self,
            single_junction_intensity_mask: BioMedicalMask,
            im_junction: BioMedicalChannel,
            contours: np.ndarray,
        ):
            """Initialize the properties with the given mask and intensity."""
            super().__init__(single_junction_intensity_mask, im_junction)

            self.intensity_cropped, self.mask_cropped = center_single_cell(
                [im_junction, single_junction_intensity_mask], contours
            )

    def __init__(
        self,
        im_junction: BioMedicalChannel,
        single_cell_mask: BioMedicalMask,
        single_cell_membrane_mask: BioMedicalMask,
        single_cell_junction_intensity_mask: BioMedicalMask,
        cue_direction: int,
        dp_epsilon: int,
    ):
        """Initialize the properties."""
        self.im_junction = im_junction
        self.single_cell_mask = single_cell_mask
        self.single_membrane_mask = single_cell_membrane_mask
        self.single_cell_junction_intensity_mask = (
            single_cell_junction_intensity_mask  # thresholded membrane mask
        )
        self.single_cell_junction_extended_mask = BioMedicalMask(
            np.logical_or(self.single_cell_mask.data, self.single_membrane_mask.data)
        )  # sc mask + sc membrane mask
        self.cue_direction = cue_direction
        self.dp_epsilon = dp_epsilon

        # specific junction properties
        self.sc_junction_interface_props = self.SingleCellJunctionInterfaceProps(
            single_cell_membrane_mask, im_junction
        )

        quadrant_masks, _partition_polygons, contours = partition_single_cell_mask(
            self.single_cell_junction_extended_mask,
            cue_direction,
            self.sc_junction_interface_props.axis_major_length,
            4,
        )
        # quadrant_masks_centered = [center_single_cell([q], p) for q, p in zip(quadrant_masks, _partition_polygons)]
        self.quadrant_masks = center_single_cell(
            quadrant_masks, contours
        )  # [item for sublist in quadrant_masks_centered for item in sublist]

        half_masks, _partition_polygons, _ = partition_single_cell_mask(
            self.single_cell_junction_extended_mask,
            cue_direction,
            self.sc_junction_interface_props.axis_major_length,
            2,
        )
        # half_masks_centered = [center_single_cell([h], p) for h, p in zip(half_masks, _partition_polygons)]
        self.half_masks = center_single_cell(
            half_masks, contours
        )  # [item for sublist in half_masks_centered for item in sublist]

        self.sc_junction_intensity_props = self.SingleCellJunctionIntensityProps(
            single_cell_junction_intensity_mask, im_junction, contours
        )

    @property
    def straight_line_junction_length(self):
        """Return the length of the straight line junction."""
        return straight_line_length(
            get_corner(self.single_cell_mask.data, self.dp_epsilon)
        )

    @property
    def interface_perimeter(self):
        """Return the perimeter of the junction interface."""
        return skimage.measure.perimeter(self.single_cell_mask.data)

    @property
    def junction_interface_linearity_index(self):
        """Return the linearity index of the junction interface."""
        return self.interface_perimeter / self.straight_line_junction_length

    @property
    def junction_interface_occupancy(self):
        """Return the occupancy of the junction interface."""
        return (
            self.sc_junction_intensity_props.area
            / self.sc_junction_interface_props.area
        )

    @property
    def junction_protein_intensity(self):
        """Return the intensity of the junction protein area."""
        return (
            self.sc_junction_intensity_props.mean_intensity
            * self.sc_junction_intensity_props.area
        )

    @property
    def junction_intensity_per_interface_area(self):
        """Return the intensity of the junction protein area and the junction interface area."""
        return self.junction_protein_intensity / self.sc_junction_interface_props.area

    @property
    def junction_cluster_density(self):
        """Return the ratio between the junction protein intensity and the junction protein area."""
        return self.junction_protein_intensity / self.sc_junction_intensity_props.area

    @property
    def junction_cue_directional_intensity_ratio(self):
        """Return the ratio of the left vs right cell membrane intensity in cue direction."""
        sc_junction_intensity_mask = BioMedicalMask.from_threshold_otsu(
            self.sc_junction_intensity_props.intensity_cropped
            * self.sc_junction_intensity_props.mask_cropped,
            gaussian_filter=None,
            rolling_ball_radius=None,  # self.sc_junction_intensity_props.intensity_cropped.shape[0] // 100,
        )
        sc_junction_intensity_mask_r = sc_junction_intensity_mask.combine(
            BioMedicalMask(self.half_masks[0])
        ).mask_background()
        sc_junction_intensity_mask_l = sc_junction_intensity_mask.combine(
            BioMedicalMask(self.half_masks[1])
        ).mask_background()

        right = (
            self.sc_junction_intensity_props.intensity_cropped
            * sc_junction_intensity_mask_r.data
        )
        left = (
            self.sc_junction_intensity_props.intensity_cropped
            * sc_junction_intensity_mask_l.data
        )

        left_m = 0 if np.ma.count_masked(left) == left.size else np.mean(left)
        right_m = 0 if np.ma.count_masked(right) == right.size else np.mean(right)

        if left_m == 0 and right_m == 0:
            warnings.warn("Warning: entire cell masked.", stacklevel=2)
            return 0

        return 1 - 2 * left_m / (left_m + right_m)

    @property
    def junction_cue_undirectional_intensity_ratio(self):
        """Return the ratio of the sum of cell membrane quarters in cue direction and the total membrane intensity."""
        sc_junction_intensity_mask = BioMedicalMask.from_threshold_otsu(
            self.sc_junction_intensity_props.intensity_cropped
            * self.sc_junction_intensity_props.mask_cropped,
            gaussian_filter=None,
            rolling_ball_radius=None,  # self.sc_junction_intensity_props.intensity_cropped.shape[0] // 100,
        )
        sc_junction_intensity_mask_l = sc_junction_intensity_mask.combine(
            BioMedicalMask(self.quadrant_masks[1])
        ).mask_background()
        sc_junction_intensity_mask_r = sc_junction_intensity_mask.combine(
            BioMedicalMask(self.quadrant_masks[3])
        ).mask_background()
        sc_junction_intensity_mask_t = sc_junction_intensity_mask.combine(
            BioMedicalMask(self.quadrant_masks[0])
        ).mask_background()
        sc_junction_intensity_mask_b = sc_junction_intensity_mask.combine(
            BioMedicalMask(self.quadrant_masks[2])
        ).mask_background()

        left = (
            self.sc_junction_intensity_props.intensity_cropped
            * sc_junction_intensity_mask_l.data
        )
        right = (
            self.sc_junction_intensity_props.intensity_cropped
            * sc_junction_intensity_mask_r.data
        )
        top = (
            self.sc_junction_intensity_props.intensity_cropped
            * sc_junction_intensity_mask_t.data
        )
        bottom = (
            self.sc_junction_intensity_props.intensity_cropped
            * sc_junction_intensity_mask_b.data
        )

        left_m = 0 if np.ma.count_masked(left) == left.size else np.mean(left)
        right_m = 0 if np.ma.count_masked(right) == right.size else np.mean(right)
        top_m = 0 if np.ma.count_masked(top) == top.size else np.mean(top)
        bottom_m = 0 if np.ma.count_masked(bottom) == bottom.size else np.mean(bottom)

        if sum([left_m, right_m, top_m, bottom_m]) == 0:
            warnings.warn("Warning: entire cell masked.", stacklevel=2)
            return 0

        return (top_m + bottom_m) / (left_m + right_m + top_m + bottom_m)


class NeighborhoodProps:
    """Class representing the properties of cell neighborhood."""

    def __init__(self):
        """Initialize the properties."""
        self.num_neighbours = None
        self.mean_dif_first_neighbors = 0
        self.median_dif_first_neighbors = 0
        self.var_dif_first_neighbors = 0
        self.range_dif_first_neighbors = 0
        self.mean_dif_second_neighbors = 0
        self.median_dif_second_neighbors = 0
        self.var_dif_second_neighbors = 0
        self.range_dif_second_neighbors = 0


class SingleCellPropertiesCollection:
    """Collection of properties of a single cell."""

    def __init__(
        self,
        single_cell_props: SingleCellProps,
        nucleus_props: Optional[SingleCellNucleusProps],
        organelle_props: Optional[SingleCellOrganelleProps],
        marker_props: Optional[SingleCellMarkerProps],
        marker_membrane_props: Optional[SingleCellMarkerMembraneProps],
        marker_nuc_props: Optional[SingleCellMarkerNucleiProps],
        marker_nuc_cyt_props: Optional[SingleCellMarkerCytosolProps],
        junction_props: Optional[SingleCellJunctionProps],
    ):
        """Initialize the properties collection."""
        self.nucleus_props = nucleus_props
        self.organelle_props = organelle_props
        self.single_cell_props = single_cell_props
        self.marker_nuc_props = marker_nuc_props
        self.marker_nuc_cyt_props = marker_nuc_cyt_props
        self.marker_membrane_props = marker_membrane_props
        self.marker_props = marker_props
        self.junction_props = junction_props
