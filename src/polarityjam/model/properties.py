from typing import Tuple

import numpy as np
import skimage.measure
from scipy import ndimage as ndi
from skimage.measure._regionprops import RegionProperties

from polarityjam import RuntimeParameter
from polarityjam.compute.compute import map_single_cell_to_circle, compute_reference_target_orientation_rad, \
    compute_angle_deg, compute_marker_vector_norm, compute_shape_orientation_rad, \
    straight_line_length
from polarityjam.compute.corner import get_corner


class SingleCellProps(RegionProperties):
    """Base class for all single cell properties."""

    def __init__(self, single_cell_mask: np.ndarray, intensity: np.ndarray = None):
        if not np.issubdtype(single_cell_mask.dtype, np.integer):
            raise RuntimeError("Only integer images allowed!")

        objects = ndi.find_objects(single_cell_mask)

        if len(objects) > 1:
            raise RuntimeError("Several objects detected in single cell mask! Aborting...")

        sl = objects[0]

        self._mask = single_cell_mask

        super().__init__(sl, 1, single_cell_mask, intensity, True)


class SingleCellCellProps(SingleCellProps):
    """Class representing the properties of a single cell."""

    def __init__(self, single_cell_mask: np.ndarray, param: RuntimeParameter):
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
        return get_corner(self._mask, self.param.dp_epsilon)


class SingleCellNucleusProps(SingleCellProps):
    """Class representing the properties of a single nucleus."""
    def __init__(self, single_nucleus_mask: np.ndarray, sc_props: SingleCellCellProps):
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


class SingleCellOrganelleProps(SingleCellProps):
    """Class representing the properties of a single organelle."""
    def __init__(self, single_organelle_mask: np.ndarray, nucleus_props: SingleCellNucleusProps):
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


class SingleCellMarkerProps(SingleCellProps):
    """Class representing the properties of a single cell marker signal."""
    def __init__(self, single_cell_mask: np.ndarray, im_marker: np.ndarray):
        super().__init__(single_cell_mask, im_marker)

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


class SingleCellMarkerMembraneProps(SingleCellProps):
    """Class representing the properties of a single cell membrane signal."""
    def __init__(self, single_membrane_mask: np.ndarray, im_marker: np.ndarray):
        super().__init__(single_membrane_mask, im_marker)

    @property
    def marker_sum_expression_mem(self):
        return self.mean_intensity * self.area


class SingleCellMarkerNucleiProps(SingleCellProps):
    """Class representing the properties of a single cell marker nucleus signal."""
    def __init__(self, single_nucleus_mask: np.ndarray, im_marker: np.ndarray, sc_nucleus_props: SingleCellNucleusProps,
                 sc_marker_props: SingleCellMarkerProps):
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


class SingleCellMarkerCytosolProps(SingleCellProps):
    """Class representing the properties of a single cell marker cytosol signal."""
    def __init__(self, single_cytosol_mask: np.ndarray, im_marker: np.ndarray,
                 sc_marker_nuclei_props: SingleCellMarkerNucleiProps):
        super().__init__(single_cytosol_mask, im_marker)
        self.sc_marker_nuclei_props = sc_marker_nuclei_props

    @property
    def marker_sum_expression_cyt(self):
        return self.mean_intensity * self.area

    @property
    def marker_cytosol_ratio(self):
        return self.sc_marker_nuclei_props.mean_intensity / self.mean_intensity


class SingleCellJunctionInterfaceProps(SingleCellProps):
    # Based on junction mapper: https://doi.org/10.7554/eLife.45413
    def __init__(self, single_membrane_mask: np.ndarray, im_junction: np.ndarray):
        super().__init__(single_membrane_mask, im_junction)


class SingleCellJunctionProteinProps(SingleCellProps):
    def __init__(self, single_junction_protein_area_mask: np.ndarray,
                 im_junction_protein_single_cell: np.ndarray):
        super().__init__(single_junction_protein_area_mask, im_junction_protein_single_cell)


class SingleCellJunctionProteinCircularProps(SingleCellProps):
    def __init__(self, im_junction_protein_single_cell: np.ndarray, cell_minor_axis_length: float,
                 interface_centroid: Tuple[float, float]):
        # todo: check
        r = cell_minor_axis_length / 2
        circular_img = map_single_cell_to_circle(im_junction_protein_single_cell, interface_centroid[0],
                                                 interface_centroid[1], r)
        circular_junction_protein_single_cell_mask = circular_img.astype(bool).astype(int)

        super().__init__(circular_junction_protein_single_cell_mask, circular_img)


class SingleCellJunctionProps:
    """Class representing the properties of a single cell junction."""
    def __init__(self, sc_junction_interface_props: SingleCellJunctionInterfaceProps,
                 sc_junction_protein_props: SingleCellJunctionProteinProps,
                 sc_junction_protein_circular_props: SingleCellJunctionProteinCircularProps,
                 sc_mask: np.ndarray, params: RuntimeParameter):
        self.sc_mask = sc_mask
        self.sc_junction_interface_props = sc_junction_interface_props
        self.sc_junction_protein_props = sc_junction_protein_props
        self.sc_junction_protein_circular_props = sc_junction_protein_circular_props
        self.params = params

    @property
    def straight_line_junction_length(self):
        return straight_line_length(get_corner(self.sc_mask, self.params.dp_epsilon))

    @property
    def interface_perimeter(self):
        return skimage.measure.perimeter(self.sc_mask)

    @property
    def junction_interface_linearity_index(self):
        return self.interface_perimeter / self.straight_line_junction_length

    @property
    def junction_interface_occupancy(self):
        return self.sc_junction_protein_props.area / self.sc_junction_interface_props.area

    @property
    def junction_protein_intensity(self):
        return self.sc_junction_protein_props.mean_intensity * self.sc_junction_protein_props.area

    @property
    def junction_intensity_per_interface_area(self):
        return self.junction_protein_intensity / self.sc_junction_interface_props.area

    @property
    def junction_cluster_density(self):
        return self.junction_protein_intensity / self.sc_junction_protein_props.area


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
    def __init__(self, single_cell_props: SingleCellCellProps,
                 nucleus_props: SingleCellNucleusProps,
                 organelle_props: SingleCellOrganelleProps,
                 marker_props: SingleCellMarkerProps,
                 marker_membrane_props: SingleCellMarkerMembraneProps,
                 marker_nuc_props: SingleCellMarkerNucleiProps,
                 marker_nuc_cyt_props: SingleCellMarkerCytosolProps,
                 junction_props: SingleCellJunctionProps):
        self.nucleus_props = nucleus_props
        self.organelle_props = organelle_props
        self.single_cell_props = single_cell_props
        self.marker_nuc_props = marker_nuc_props
        self.marker_nuc_cyt_props = marker_nuc_cyt_props
        self.marker_membrane_props = marker_membrane_props
        self.marker_props = marker_props
        self.junction_props = junction_props
