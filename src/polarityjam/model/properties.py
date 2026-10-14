"""Hold all classes and functions related to the properties of a cell."""
from __future__ import annotations

import warnings
from functools import cached_property
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import skimage.measure
from scipy import ndimage as ndi
from shapely.geometry import Polygon
from skimage.measure._regionprops import RegionProperties

from polarityjam.compute.compute import (
    compute_angle_deg,
    compute_marker_vector_norm,
    compute_reference_target_orientation_rad,
    compute_shape_orientation_rad,
    straight_line_length,
)
from polarityjam.compute.corner import get_contour, get_corner
from polarityjam.compute.shape import (
    get_cue_side,
    get_linear_fit,
    get_max_nucleus_membrane_distance,
    get_nucleus_membrane_distance,
    get_radial_intensity_profile,
    get_radial_position,
    get_shanon_estimated_pdf_center_distance,
    mirror_along_cue_direction,
)
from polarityjam.model.masks import BioMedicalMask
from polarityjam.polarityjam_logging import get_logger

if TYPE_CHECKING:
    from polarityjam.model.image import BioMedicalChannel


class SingleInstanceProps(RegionProperties):
    """Base class for all single cell properties."""

    def __init__(
        self,
        single_cell_mask: BioMedicalMask,
        intensity: Optional[BioMedicalChannel] = None,
    ):
        """Initialize the properties with the given mask and intensity."""
        _intensity = None
        if intensity is not None:
            _intensity = intensity.data

        objects = ndi.find_objects(single_cell_mask.data.astype(np.uint8))

        assert len(objects) == 1, "Single cell mask should contain exactly one object!"

        sl = objects[0]

        self.mask = single_cell_mask
        self.intensity = intensity

        super().__init__(
            sl, 1, single_cell_mask.data, intensity_image=_intensity, cache_active=True
        )


class SingleCellProps(SingleInstanceProps):
    """Class representing the properties of a single cell."""

    def __init__(
        self,
        single_cell_mask: BioMedicalMask,
        single_cell_centered_mask: BioMedicalMask,
        dp_epsilon: int,
        cue_direction: int,
    ):
        """Initialize the properties with the given mask and intensity."""
        self.dp_epsilon = dp_epsilon
        self.cue_direction = cue_direction
        self.single_cell_centered_mask = single_cell_centered_mask
        super().__init__(single_cell_mask)

    @property
    def cell_cue_direction_symmetry(self):
        """Return the asymmetry of the cell."""
        mask_mirror_cue_direction_up = mirror_along_cue_direction(
            self.single_cell_centered_mask.data, (self.cue_direction + 90) % 360
        )
        mask_mirror_cue_direction_down = mirror_along_cue_direction(
            self.single_cell_centered_mask.data, (self.cue_direction + 270) % 360
        )

        # calculate IoU between mask_mirror_cue_direction_up and mask_mirror_cue_direction_down
        iou = np.sum(
            np.logical_and(mask_mirror_cue_direction_up, mask_mirror_cue_direction_down)
        ) / np.sum(
            np.logical_or(mask_mirror_cue_direction_up, mask_mirror_cue_direction_down)
        )

        return iou

    @property
    def center_distance_entropy(self):
        """Return the entropy of the distance of the cell center to the cell border."""
        contours = get_contour(self.single_cell_centered_mask.data)

        pg = Polygon(contours)  # swaps x and y

        pg_cent_a = int(pg.centroid.coords.xy[0][0])
        pg_cent_b = int(pg.centroid.coords.xy[1][0])

        origin = np.array([pg_cent_b, pg_cent_a], dtype="float")

        # calculate distance of each pixel to the center
        e = get_shanon_estimated_pdf_center_distance(origin, contours)

        return e

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
        warnings.warn(
            "Use cell_length_to_width_ratio instead.", DeprecationWarning, stacklevel=2
        )
        return self.cell_length_to_width_ratio

    @property
    def cell_length_to_width_ratio(self):
        """Return the major to minor ratio of the cell."""
        return self.major_axis_length / self.minor_axis_length

    @property
    def cell_corner_points(self):
        """Return the corner points of the cell."""
        return get_corner(self.mask.data, self.dp_epsilon)


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
        warnings.warn(
            "Use nuc_length_to_width_ratio instead.", DeprecationWarning, stacklevel=2
        )
        return self.nuc_length_to_width_ratio

    @property
    def nuc_length_to_width_ratio(self):
        """Return the major to minor ratio of the nucleus."""
        if self.minor_axis_length == 0:
            get_logger().warning("Warning: Minor axis length is zero.", stacklevel=2)
            return np.inf

        return self.major_axis_length / self.minor_axis_length

    @property
    def contour_points(self):
        """Return the contour points of the nucleus."""
        return get_contour(self.mask.data)

    @property
    def nuc_displacement_distance(self):
        """Return the distance from nucleus center to cell center."""
        return compute_marker_vector_norm(
            self._sc_props.centroid[0],  # single cell center X
            self._sc_props.centroid[1],
            self.centroid[0],  # nucleus center X
            self.centroid[1],
        )

    @property
    def nuc_membrane_distance(self):
        """Return the shortest distance between the nucleus and the cell membrane."""
        return get_nucleus_membrane_distance(
            self._sc_props.image, self.mask.data[self._sc_props.slice]
        )

    @property
    def nuc_membrane_position_index(self):
        """Return the nucleus membrane distance relative to the largest one possible in the cell."""
        max_distance = get_max_nucleus_membrane_distance(
            self._sc_props.image, self.mask.data[self._sc_props.slice]
        )

        if max_distance == 0:
            get_logger().warning(
                "Warning: Nucleus does not fit into the cell without touching the membrane.",
                stacklevel=2,
            )
            return np.nan

        return self.nuc_membrane_distance / max_distance


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
        return get_contour(self.mask.data)


class SingleCellMarkerProps(SingleInstanceProps):
    """Class representing the properties of a single cell marker signal."""

    def __init__(
        self,
        single_cell_mask: BioMedicalMask,
        im_marker: BioMedicalChannel,
        quadrant_cell_masks: List[BioMedicalMask],
        half_cell_masks: List[BioMedicalMask],
    ):
        """Initialize the properties with the given mask and intensity."""
        super().__init__(single_cell_mask, im_marker)
        self.quadrant_masks = quadrant_cell_masks
        self.half_masks = half_cell_masks

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

    def _99_percentile_thresh(self):
        """Return the 99th percentile of the array."""
        # single cell intensity
        sc_marker_intensity = self.intensity.data * self.mask.data

        # 99th percentile of the single cell marker intensity
        sc_marker_intensity_masked = (
            self.intensity.data * self.mask.mask_background().data
        )
        sc_marker_intensity_masked = np.ma.filled(
            sc_marker_intensity_masked.astype(float), np.nan
        )
        sc_marker_intensity_99 = np.nanpercentile(sc_marker_intensity_masked, 99)

        # threshold the single cell marker intensity
        sc_marker_intensity = np.where(
            sc_marker_intensity > sc_marker_intensity_99,
            sc_marker_intensity_99,
            sc_marker_intensity,
        )

        return sc_marker_intensity

    @property
    def marker_cue_directional_intensity_ratio(self):
        """Return the ratio of the left vs right cell marker intensity in cue direction."""
        # single cell intensity
        sc_marker_intensity = self._99_percentile_thresh()

        # compute an intensity mask
        sc_marker_intensity_mask = BioMedicalMask.from_threshold_otsu(
            sc_marker_intensity,
            gaussian_filter=None,
            rolling_ball_radius=None,
        )
        if len(self.half_masks) > 1:
            sc_marker_intensity_mask_r = sc_marker_intensity_mask.combine(
                self.half_masks[0]
            ).mask_background()
            sc_marker_intensity_mask_l = sc_marker_intensity_mask.combine(
                self.half_masks[1]
            ).mask_background()
        else:
            warnings.warn(
                "Warning: Number of partitions(2) does not match the number of created masks (1):",
                stacklevel=2,
            )
            return np.nan

        right = (
            sc_marker_intensity_mask_r.data * sc_marker_intensity
        )  # masked right half
        left = sc_marker_intensity_mask_l.data * sc_marker_intensity  # masked left half

        left_m = 0 if np.ma.count_masked(left) == left.size else np.ma.mean(left)
        right_m = 0 if np.ma.count_masked(right) == right.size else np.ma.mean(right)

        if left_m == 0 and right_m == 0:
            warnings.warn("Warning: entire cell masked.", stacklevel=2)
            return 0

        return 1 - 2 * left_m / (left_m + right_m)

    @property
    def marker_cue_axial_intensity_ratio(self):
        """The ratio of the sum of cell marker quarters in cue direction and the total marker intensity."""
        # single cell intensity
        sc_marker_intensity = self._99_percentile_thresh()

        sc_marker_intensity_mask = BioMedicalMask.from_threshold_otsu(
            sc_marker_intensity,
            gaussian_filter=None,
            rolling_ball_radius=None,
        )
        if len(self.quadrant_masks) > 3:
            sc_marker_intensity_mask_r = sc_marker_intensity_mask.combine(
                self.quadrant_masks[0]
            ).mask_background()
            sc_marker_intensity_mask_t = sc_marker_intensity_mask.combine(
                self.quadrant_masks[1]
            ).mask_background()
            sc_marker_intensity_mask_l = sc_marker_intensity_mask.combine(
                self.quadrant_masks[2]
            ).mask_background()
            sc_marker_intensity_mask_b = sc_marker_intensity_mask.combine(
                self.quadrant_masks[3]
            ).mask_background()
        else:
            warnings.warn(
                "Warning: Number of partitions(4) does not match the number of created masks (3):",
                stacklevel=2,
            )
            return np.nan

        top = sc_marker_intensity_mask_t.data * sc_marker_intensity  # masked top half
        left = sc_marker_intensity_mask_l.data * sc_marker_intensity  # masked left half
        bottom = (
            sc_marker_intensity_mask_b.data * sc_marker_intensity
        )  # masked bottom half
        right = (
            sc_marker_intensity_mask_r.data * sc_marker_intensity
        )  # masked right half

        left_m = 0 if np.ma.count_masked(left) == left.size else np.ma.mean(left)
        right_m = 0 if np.ma.count_masked(right) == right.size else np.ma.mean(right)
        top_m = 0 if np.ma.count_masked(top) == top.size else np.ma.mean(top)
        bottom_m = (
            0 if np.ma.count_masked(bottom) == bottom.size else np.ma.mean(bottom)
        )

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
        single_cell_mask: BioMedicalMask,
        single_membrane_mask: BioMedicalMask,
        cue_direction: int,
        num_radial_rings: int,
    ):
        """Initialize the properties with the given mask and intensity."""
        super().__init__(single_cytosol_mask, im_marker)

        if num_radial_rings < 2:
            raise ValueError(
                "At least two radial rings are needed to fit the marker radial gradient."
            )

        self.sc_marker_nuclei_props = sc_marker_nuclei_props
        self.cell_mask = single_cell_mask
        self.membrane_mask = single_membrane_mask
        self.cue_direction = cue_direction
        # number of rings of the radial intensity profile from the nucleus to the membrane
        self.num_radial_rings = num_radial_rings

    @property
    def marker_sum_expression_cyt(self):
        """Return the sum of the marker cytosol expression."""
        return self.mean_intensity * self.area

    @property
    def marker_cytosol_ratio(self):
        """Return the ratio of the marker cytosol expression and the marker nucleus mean intensity."""
        return self.sc_marker_nuclei_props.mean_intensity / self.mean_intensity

    @cached_property
    def _radial_pixels(self):
        """Return radial position, nucleus distance, cue side and intensity of the cytosol pixels without membrane."""
        cell_mask = self.cell_mask.data.astype(bool)
        nucleus_mask = self.sc_marker_nuclei_props.mask.data.astype(bool)

        # crop to cell and nucleus, the nucleus might reach beyond the cell
        sl = ndi.find_objects((cell_mask | nucleus_mask).astype(np.uint8))[0]
        nucleus_mask = nucleus_mask[sl]

        cytosol_mask, nucleus_distance, position = get_radial_position(
            cell_mask[sl], nucleus_mask, self.membrane_mask.data[sl]
        )

        if not cytosol_mask.any():
            get_logger().warning(
                "Warning: No cytosol between nucleus and membrane.", stacklevel=2
            )

        cue_side = get_cue_side(
            cytosol_mask, ndi.center_of_mass(nucleus_mask), self.cue_direction
        )
        intensity = self.intensity.data[sl][cytosol_mask].astype(float)

        return position, nucleus_distance, cue_side, intensity

    def _radial_profile(self, pixels: Optional[np.ndarray] = None) -> np.ndarray:
        """Return the mean marker intensity of the rings from the nucleus to the membrane."""
        position, _, _, intensity = self._radial_pixels

        # all cytosol pixels if no selection is given
        if pixels is not None:
            position, intensity = position[pixels], intensity[pixels]

        return get_radial_intensity_profile(
            position, intensity, np.linspace(0, 1, self.num_radial_rings + 1)
        )

    def _radial_gradient(
        self, pixels: Optional[np.ndarray] = None
    ) -> Tuple[float, float, float]:
        """Return slope, intercept and R² of a line fitted to the radial intensity profile."""
        ring_centers = (np.arange(self.num_radial_rings) + 0.5) / self.num_radial_rings

        return get_linear_fit(ring_centers, self._radial_profile(pixels))

    def _normalize_radial_gradient(self, gradient: float) -> float:
        """Return the gradient relative to the mean intensity of the radial profile."""
        profile = self._radial_profile()

        if np.isnan(profile).all() or np.nanmean(profile) == 0:
            return np.nan

        return gradient / np.nanmean(profile)

    @property
    def marker_radial_gradient(self):
        """Return the change of the marker intensity from the nucleus to the membrane."""
        return self._radial_gradient()[0]

    @property
    def marker_radial_gradient_norm(self):
        """Return the marker radial gradient relative to the mean marker intensity."""
        return self._normalize_radial_gradient(self.marker_radial_gradient)

    @property
    def marker_radial_gradient_r2(self):
        """Return how well the radial marker intensity profile follows its gradient."""
        return self._radial_gradient()[2]

    @property
    def marker_radial_gradient_px(self):
        """Return the change of the marker intensity per pixel distance from the nucleus."""
        _, nucleus_distance, _, intensity = self._radial_pixels

        if nucleus_distance.size == 0:
            return np.nan

        # rings of one pixel width
        ring_edges = np.arange(np.ceil(nucleus_distance.max()) + 1)
        profile = get_radial_intensity_profile(nucleus_distance, intensity, ring_edges)

        return get_linear_fit(ring_edges[:-1] + 0.5, profile)[0]

    @property
    def marker_radial_intensity_ratio(self):
        """Return the marker intensity of the outer third of the cytosol relative to the inner third."""
        position, _, _, intensity = self._radial_pixels
        inner = intensity[position < 1 / 3]
        outer = intensity[position >= 2 / 3]

        if inner.size == 0 or outer.size == 0 or inner.mean() == 0:
            return np.nan

        return outer.mean() / inner.mean()

    @property
    def marker_radial_gradient_norm_cue_front(self):
        """Return the normalized marker radial gradient in front of the nucleus in cue direction."""
        _, _, cue_side, _ = self._radial_pixels

        return self._normalize_radial_gradient(self._radial_gradient(cue_side > 0)[0])

    @property
    def marker_radial_gradient_norm_cue_back(self):
        """Return the normalized marker radial gradient behind the nucleus in cue direction."""
        _, _, cue_side, _ = self._radial_pixels

        return self._normalize_radial_gradient(self._radial_gradient(cue_side < 0)[0])


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
        ):
            """Initialize the properties with the given mask and intensity."""
            super().__init__(single_junction_intensity_mask, im_junction)

            self.intensity = im_junction
            self.mask = single_junction_intensity_mask

    def __init__(
        self,
        im_junction: BioMedicalChannel,
        single_cell_mask: BioMedicalMask,
        single_cell_membrane_mask: BioMedicalMask,
        single_cell_junction_intensity_mask: BioMedicalMask,
        single_cell_props: SingleCellProps,
        quadrant_masks: List[BioMedicalMask],
        half_masks: List[BioMedicalMask],
        cue_direction: int,
        dp_epsilon: int,
    ):
        """Initialize the properties."""
        self.im_junction = im_junction
        self.single_cell_mask = single_cell_mask
        self.single_membrane_mask = single_cell_membrane_mask
        self.single_cell_junction_intensity_mask = (
            single_cell_junction_intensity_mask  # threshold membrane mask
        )
        self.single_cell_junction_extended_mask = BioMedicalMask(
            np.logical_or(self.single_cell_mask.data, self.single_membrane_mask.data)
        )  # sc mask + sc membrane mask

        self.single_cell_props = single_cell_props

        self.quadrant_masks = quadrant_masks
        self.half_masks = half_masks

        self.cue_direction = cue_direction
        self.dp_epsilon = dp_epsilon

        # specific junction properties
        self.sc_junction_interface_props = self.SingleCellJunctionInterfaceProps(
            single_cell_membrane_mask, im_junction
        )

        self.sc_junction_intensity_props = self.SingleCellJunctionIntensityProps(
            single_cell_junction_intensity_mask, im_junction
        )

    @property
    def junction_centroid_orientation_rad(self):
        """Return the orientation of the marker nucleus in radians."""
        return compute_reference_target_orientation_rad(
            self.single_cell_props.centroid[0],
            self.single_cell_props.centroid[1],
            self.sc_junction_intensity_props.weighted_centroid[0],
            self.sc_junction_intensity_props.weighted_centroid[1],
        )

    @property
    def junction_centroid_orientation_deg(self):
        """Return the orientation of the marker nucleus in degrees."""
        return compute_angle_deg(self.junction_centroid_orientation_rad)

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
        # TODO: revise this part of the code
        if len(self.quadrant_masks) == 4:
            sc_junction_intensity_mask_r = (
                self.single_cell_junction_intensity_mask.combine(
                    self.half_masks[0]
                    .combine(self.single_membrane_mask)
                    .mask_background()
                )
            )
            sc_junction_intensity_mask_l = (
                self.single_cell_junction_intensity_mask.combine(
                    self.half_masks[1]
                    .combine(self.single_membrane_mask)
                    .mask_background()
                )
            )
        else:
            warnings.warn(
                "Warning: Number of partitions(2) does not match the number of created masks (1):",
                stacklevel=2,
            )
            return np.nan

        right = (
            self.sc_junction_intensity_props.intensity.data
            * sc_junction_intensity_mask_r.data
        )
        left = (
            self.sc_junction_intensity_props.intensity.data
            * sc_junction_intensity_mask_l.data
        )

        left_m = 0 if np.ma.count_masked(left) == left.size else np.ma.mean(left)
        right_m = 0 if np.ma.count_masked(right) == right.size else np.ma.mean(right)

        if left_m == 0 and right_m == 0:
            warnings.warn(
                "Warning: entire cell masked. Consider using a different thresholding method.",
                stacklevel=2,
            )
            return 0

        return 1 - 2 * left_m / (left_m + right_m)

    @property
    def junction_cue_axial_intensity_ratio(self):
        """Return the ratio of the sum of cell membrane quarters in cue direction and the total membrane intensity."""
        # TODO: revise this part of the code
        if len(self.quadrant_masks) == 4:
            sc_junction_intensity_mask_r = (
                self.single_cell_junction_intensity_mask.combine(
                    self.quadrant_masks[0]
                    .combine(self.single_membrane_mask)
                    .mask_background()
                )
            )
            sc_junction_intensity_mask_t = (
                self.single_cell_junction_intensity_mask.combine(
                    self.quadrant_masks[1]
                    .combine(self.single_membrane_mask)
                    .mask_background()
                )
            )
            sc_junction_intensity_mask_l = (
                self.single_cell_junction_intensity_mask.combine(
                    self.quadrant_masks[2]
                    .combine(self.single_membrane_mask)
                    .mask_background()
                )
            )
            sc_junction_intensity_mask_b = (
                self.single_cell_junction_intensity_mask.combine(
                    self.quadrant_masks[3]
                    .combine(self.single_membrane_mask)
                    .mask_background()
                )
            )
        else:
            warnings.warn(
                "Warning: Number of partitions(4) does not match the number of created masks (3):",
                stacklevel=2,
            )
            return np.nan

        left = (
            self.sc_junction_intensity_props.intensity.data
            * sc_junction_intensity_mask_l.data
        )
        right = (
            self.sc_junction_intensity_props.intensity.data
            * sc_junction_intensity_mask_r.data
        )
        top = (
            self.sc_junction_intensity_props.intensity.data
            * sc_junction_intensity_mask_t.data
        )
        bottom = (
            self.sc_junction_intensity_props.intensity.data
            * sc_junction_intensity_mask_b.data
        )

        left_m = 0 if np.ma.count_masked(left) == left.size else np.ma.mean(left)
        right_m = 0 if np.ma.count_masked(right) == right.size else np.ma.mean(right)
        top_m = 0 if np.ma.count_masked(top) == top.size else np.ma.mean(top)
        bottom_m = (
            0 if np.ma.count_masked(bottom) == bottom.size else np.ma.mean(bottom)
        )

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
        nucleus_props: Optional[SingleCellNucleusProps] = None,
        organelle_props: Optional[SingleCellOrganelleProps] = None,
        marker_props: Optional[SingleCellMarkerProps] = None,
        marker_membrane_props: Optional[SingleCellMarkerMembraneProps] = None,
        marker_nuc_props: Optional[SingleCellMarkerNucleiProps] = None,
        marker_nuc_cyt_props: Optional[SingleCellMarkerCytosolProps] = None,
        junction_props: Optional[SingleCellJunctionProps] = None,
    ):
        """Initialize the properties' collection."""
        self.nucleus_props = nucleus_props
        self.organelle_props = organelle_props
        self.single_cell_props = single_cell_props
        self.marker_nuc_props = marker_nuc_props
        self.marker_nuc_cyt_props = marker_nuc_cyt_props
        self.marker_membrane_props = marker_membrane_props
        self.marker_props = marker_props
        self.junction_props = junction_props
