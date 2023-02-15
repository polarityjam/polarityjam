from __future__ import annotations

from typing import Union

import numpy as np
import skimage.measure
from scipy import ndimage as ndi
from skimage.measure._regionprops import RegionProperties

from polarityjam.model.parameter import RuntimeParameter
from polarityjam.model.masks import BioMedicalMask
from polarityjam.compute.compute import compute_reference_target_orientation_rad, \
    compute_angle_deg, compute_marker_vector_norm, compute_shape_orientation_rad, \
    straight_line_length
from polarityjam.compute.corner import get_corner, get_contour
from polarityjam.compute.shape import partition_single_cell_mask
from polarityjam.model.image import BioMedicalChannel


class SingleInstanceProps(RegionProperties):
    """Base class for all single cell properties."""

    def __init__(
            self, single_cell_mask: Union[np.ndarray, BioMedicalMask],
            intensity: Union[np.ndarray, BioMedicalChannel] = None
    ):

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
        self.param = param
        super().__init__(single_cell_mask)

    @property
    def cell_shape_orientation_rad(self):
        # note, the values of orientation from props are in [-pi/2,pi/2] with zero along the y-axis
        return compute_shape_orientation_rad(self.orientation)

    @property
    def cell_shape_orientation_deg(self):
        # note, the values of orientation from props are in [-pi/2,pi/2] with zero along the y-axis
        return compute_angle_deg(self.cell_shape_orientation_rad)

    @property
    def cell_major_to_minor_ratio(self):
        return self.major_axis_length / self.minor_axis_length

    @property
    def cell_corner_points(self):
        return get_corner(self.mask, self.param.dp_epsilon)


class SingleCellNucleusProps(SingleInstanceProps):
    """Class representing the properties of a single nucleus."""

    def __init__(self, single_nucleus_mask: BioMedicalMask, sc_props: SingleCellProps):
        super().__init__(single_nucleus_mask)

        self._sc_props = sc_props

    @property
    def nuc_displacement_orientation_rad(self):
        return compute_reference_target_orientation_rad(
            self._sc_props.centroid[0], self._sc_props.centroid[1], self.centroid[0], self.centroid[1]
        )

    @property
    def nuc_displacement_orientation_deg(self):
        return compute_angle_deg(self.nuc_displacement_orientation_rad)

    @property
    def nuc_shape_orientation_rad(self):
        # note, the values of orientation from props are in [-pi/2,pi/2] with zero along the y-axis
        return compute_shape_orientation_rad(self.orientation)

    @property
    def nuc_shape_orientation_deg(self):
        return compute_angle_deg(self.nuc_shape_orientation_rad)

    @property
    def nuc_major_to_minor_ratio(self):
        return self.major_axis_length / self.minor_axis_length

    @property
    def contour_points(self):
        return get_contour(self.mask)


class SingleCellOrganelleProps(SingleInstanceProps):
    """Class representing the properties of a single organelle."""

    def __init__(self, single_organelle_mask: BioMedicalMask, nucleus_props: SingleCellNucleusProps):
        super().__init__(single_organelle_mask)

        self._nucleus_props = nucleus_props

    @property
    def nuc_organelle_distance(self):
        return compute_marker_vector_norm(
            self.centroid[0], self.centroid[1], self._nucleus_props.centroid[0], self._nucleus_props.centroid[1]
        )

    @property
    def organelle_orientation_rad(self):
        return compute_reference_target_orientation_rad(
            self._nucleus_props.centroid[0], self._nucleus_props.centroid[1], self.centroid[0], self.centroid[1]
        )

    @property
    def organelle_orientation_deg(self):
        return compute_angle_deg(self.organelle_orientation_rad)

    @property
    def contour_points(self):
        return get_contour(self.mask)


class SingleCellMarkerProps(SingleInstanceProps):
    """Class representing the properties of a single cell marker signal."""

    def __init__(self, single_cell_mask: BioMedicalMask, im_marker: BioMedicalChannel, cue_direction: int):
        super().__init__(single_cell_mask, im_marker)
        self.quadrant_masks, self._partition_polygons = partition_single_cell_mask(
            single_cell_mask.data, cue_direction, self.axis_major_length, 4
        )
        self.half_masks, self._partition_polygons = partition_single_cell_mask(
            single_cell_mask.data, cue_direction, self.axis_major_length, 2
        )

    @property
    def marker_centroid_orientation_rad(self):
        return compute_reference_target_orientation_rad(
            self.centroid[0], self.centroid[1], self.weighted_centroid[0], self.weighted_centroid[1]
        )

    @property
    def marker_centroid_orientation_deg(self):
        return compute_angle_deg(self.marker_centroid_orientation_rad)

    @property
    def marker_sum_expression(self):
        return self.mean_intensity * self.area

    @property
    def marker_cue_directional_intensity_ratio(self):
        left = self.intensity * self.half_masks[0] * self.mask
        right = self.intensity * self.half_masks[1] * self.mask

        left_m = np.mean(left)
        right_m = np.mean(right)

        return 1 - 2*left_m/(left_m + right_m)

    @property
    def marker_cue_undirectional_intensity_ratio(self):
        top = self.intensity * self.quadrant_masks[0] * self.mask
        left = self.intensity * self.quadrant_masks[1] * self.mask
        bottom = self.intensity * self.quadrant_masks[2] * self.mask
        right = self.intensity * self.quadrant_masks[3] * self.mask

        return (np.mean(top) + np.mean(bottom)) / (np.mean(left) + np.mean(right) + np.mean(top) + np.mean(bottom))


class SingleCellMarkerMembraneProps(SingleInstanceProps):
    """Class representing the properties of a single cell membrane signal."""

    def __init__(self, single_membrane_mask: BioMedicalMask, im_marker: BioMedicalChannel):
        super().__init__(single_membrane_mask, im_marker)

    @property
    def marker_sum_expression_mem(self):
        return self.mean_intensity * self.area


class SingleCellMarkerNucleiProps(SingleInstanceProps):
    """Class representing the properties of a single cell marker nucleus signal."""

    def __init__(
            self,
            single_nucleus_mask: BioMedicalMask,
            im_marker: BioMedicalChannel,
            sc_nucleus_props: SingleCellNucleusProps,
            sc_marker_props: SingleCellMarkerProps
    ):
        super().__init__(single_nucleus_mask, im_marker)
        self._sc_nucleus_props = sc_nucleus_props
        self._sc_marker_props = sc_marker_props

    @property
    def marker_nucleus_orientation_rad(self):
        return compute_reference_target_orientation_rad(
            self._sc_nucleus_props.centroid[0],
            self._sc_nucleus_props.centroid[1],
            self._sc_marker_props.centroid[0],
            self._sc_marker_props.centroid[1]
        )

    @property
    def marker_nucleus_orientation_deg(self):
        return compute_angle_deg(self.marker_nucleus_orientation_rad)

    @property
    def marker_sum_expression_nuc(self):
        return self.mean_intensity * self.area


class SingleCellMarkerCytosolProps(SingleInstanceProps):
    """Class representing the properties of a single cell marker cytosol signal."""

    def __init__(
            self,
            single_cytosol_mask: BioMedicalMask,
            im_marker: BioMedicalChannel,
            sc_marker_nuclei_props: SingleCellMarkerNucleiProps
    ):
        super().__init__(single_cytosol_mask, im_marker)
        self.sc_marker_nuclei_props = sc_marker_nuclei_props

    @property
    def marker_sum_expression_cyt(self):
        return self.mean_intensity * self.area

    @property
    def marker_cytosol_ratio(self):
        return self.sc_marker_nuclei_props.mean_intensity / self.mean_intensity


class SingleCellJunctionProps:
    # Junction properties subclasses
    class SingleCellJunctionInterfaceProps(SingleInstanceProps):
        # Based on junction mapper: https://doi.org/10.7554/eLife.45413
        def __init__(self, single_membrane_mask: BioMedicalMask, im_junction: BioMedicalChannel):
            super().__init__(single_membrane_mask, im_junction)

    class SingleCellJunctionIntensityProps(SingleInstanceProps):
        def __init__(self, single_junction_intensity_mask: BioMedicalMask, im_junction: BioMedicalChannel):
            super().__init__(single_junction_intensity_mask, im_junction)

    """Class representing the properties of a single cell junction."""

    def __init__(
            self,
            im_junction: BioMedicalChannel,
            single_cell_mask: BioMedicalMask,
            single_cell_membrane_mask: BioMedicalMask,
            single_cell_junction_intensity_mask: BioMedicalMask,
            cue_direction: int,
            dp_epsilon: int,
    ):
        self.im_junction = im_junction
        self.single_cell_mask = single_cell_mask
        self.single_membrane_mask = single_cell_membrane_mask
        self.single_cell_junction_intensity_mask = single_cell_junction_intensity_mask
        self.cue_direction = cue_direction
        self.dp_epsilon = dp_epsilon

        # specific junction properties
        self.sc_junction_interface_props = self.SingleCellJunctionInterfaceProps(single_cell_membrane_mask, im_junction)
        self.sc_junction_intensity_props = self.SingleCellJunctionIntensityProps(
            single_cell_junction_intensity_mask, im_junction
        )

        self.quadrant_masks, self._partition_polygons = partition_single_cell_mask(
            np.logical_or(self.single_cell_mask.data, self.sc_junction_interface_props.mask.data),
            cue_direction,
            self.sc_junction_interface_props.axis_major_length,
            4
        )
        self.half_masks, self._partition_polygons = partition_single_cell_mask(
            np.logical_or(self.single_cell_mask.data, self.sc_junction_interface_props.mask.data),
            cue_direction,
            self.sc_junction_interface_props.axis_major_length,
            2
        )

    @property
    def straight_line_junction_length(self):
        return straight_line_length(get_corner(self.single_cell_mask.data, self.dp_epsilon))

    @property
    def interface_perimeter(self):
        return skimage.measure.perimeter(self.single_cell_mask.data)

    @property
    def junction_interface_linearity_index(self):
        return self.interface_perimeter / self.straight_line_junction_length

    @property
    def junction_interface_occupancy(self):
        return self.sc_junction_intensity_props.area / self.sc_junction_interface_props.area

    @property
    def junction_protein_intensity(self):
        return self.sc_junction_intensity_props.mean_intensity * self.sc_junction_intensity_props.area

    @property
    def junction_intensity_per_interface_area(self):
        return self.junction_protein_intensity / self.sc_junction_interface_props.area

    @property
    def junction_cluster_density(self):
        return self.junction_protein_intensity / self.sc_junction_intensity_props.area

    @property
    def junction_cue_directional_intensity_ratio(self):
        left = self.sc_junction_intensity_props.intensity * self.half_masks[0] * \
               self.sc_junction_intensity_props.mask
        right = self.sc_junction_intensity_props.intensity * self.half_masks[1] * \
                self.sc_junction_intensity_props.mask

        left_m = np.mean(left)
        right_m = np.mean(right)

        return 1 - 2 * left_m / (left_m + right_m)

    @property
    def junction_cue_undirectional_intensity_ratio(self):
        left = self.sc_junction_intensity_props.intensity * \
               self.quadrant_masks[1] * self.sc_junction_intensity_props.mask
        right = self.sc_junction_intensity_props.intensity * \
                self.quadrant_masks[3] * self.sc_junction_intensity_props.mask
        top = self.sc_junction_intensity_props.intensity * \
              self.quadrant_masks[0] * self.sc_junction_intensity_props.mask
        bottom = self.sc_junction_intensity_props.intensity * \
                 self.quadrant_masks[2] * self.sc_junction_intensity_props.mask

        return (np.mean(top) + np.mean(bottom)) / (np.mean(left) + np.mean(right) + np.mean(top) + np.mean(bottom))


class NeighborhoodProps:
    """Class representing the properties of cell neighborhood."""

    def __init__(self):
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
            self, single_cell_props: SingleCellProps,
            nucleus_props: SingleCellNucleusProps,
            organelle_props: SingleCellOrganelleProps,
            marker_props: SingleCellMarkerProps,
            marker_membrane_props: SingleCellMarkerMembraneProps,
            marker_nuc_props: SingleCellMarkerNucleiProps,
            marker_nuc_cyt_props: SingleCellMarkerCytosolProps,
            junction_props: SingleCellJunctionProps
    ):
        self.nucleus_props = nucleus_props
        self.organelle_props = organelle_props
        self.single_cell_props = single_cell_props
        self.marker_nuc_props = marker_nuc_props
        self.marker_nuc_cyt_props = marker_nuc_cyt_props
        self.marker_membrane_props = marker_membrane_props
        self.marker_props = marker_props
        self.junction_props = junction_props
