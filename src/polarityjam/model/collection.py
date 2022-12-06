import json

import numpy as np
import pandas as pd


class PropertiesCollection:
    def __init__(self):
        self.dataset = pd.DataFrame()
        self.masks_dict = {}
        self.out_path_dict = {}
        self.img_channel_dict = {}
        self._index = 1
        self._reset_index = 1

    def current_index(self):
        return self._index

    def increase_index(self):
        self._index += 1

    def set_reset_index(self):
        self._reset_index = self._index

    def reset_index(self):
        self._index = self._reset_index

    def get_image_channel_by_img_name(self, img_name, channel) -> np.ndarray:
        return self.img_channel_dict[img_name][channel]

    def get_mask_by_img_name(self, img_name) -> np.ndarray:
        return self.masks_dict[img_name]

    def get_properties_by_img_name(self, img_name):
        return self.dataset.loc[self.dataset["filename"] == img_name]

    def get_out_path_by_name(self, img_name):
        return self.out_path_dict[img_name]

    def add_sc_marker_polarity_props(self, props):
        """Fills the dataset with the single cell marker properties."""
        self.dataset.at[self._index, "marker_mean_expression"] = props.mean_intensity
        self.dataset.at[
            self._index, "marker_sum_expression"] = props.marker_sum_expression
        self.dataset.at[self._index, "marker_centroid_X"] = props.weighted_centroid[0]
        self.dataset.at[self._index, "marker_centroid_Y"] = props.weighted_centroid[1]
        self.dataset.at[self._index, "marker_centroid_orientation_rad"] = props.marker_centroid_orientation_rad
        self.dataset.at[self._index, "marker_centroid_orientation_deg"] = props.marker_centroid_orientation_deg

    def add_sc_nucleus_props(self, props):
        """Fills the dataset with the single cell nucleus properties."""
        self.dataset.at[self._index, "nuc_X"] = props.centroid[0]
        self.dataset.at[self._index, "nuc_Y"] = props.centroid[1]
        self.dataset.at[self._index, "nuc_displacement_orientation_rad"] = props.nuc_displacement_orientation_rad
        self.dataset.at[self._index, "nuc_displacement_orientation_deg"] = props.nuc_displacement_orientation_deg
        self.dataset.at[self._index, "nuc_shape_orientation_rad"] = props.nuc_shape_orientation_rad
        self.dataset.at[self._index, "nuc_shape_orientation_deg"] = props.nuc_shape_orientation_deg
        self.dataset.at[self._index, "nuc_major_axis_length"] = props.major_axis_length
        self.dataset.at[self._index, "nuc_minor_axis_length"] = props.minor_axis_length
        self.dataset.at[self._index, "nuc_area"] = props.area
        self.dataset.at[self._index, "nuc_perimeter"] = props.perimeter
        self.dataset.at[self._index, "nuc_eccentricity"] = props.eccentricity
        self.dataset.at[self._index, "nuc_major_to_minor_ratio"] = props.nuc_major_to_minor_ratio

    def add_sc_general_props(self, filename, img_hash, connected_component_label, props):
        """Fills the dataset with the general single cell properties."""
        self.dataset.at[self._index, "filename"] = filename
        self.dataset.at[self._index, "img_hash"] = img_hash
        self.dataset.at[self._index, "label"] = connected_component_label
        self.dataset.at[self._index, "cell_X"] = props.centroid[0]
        self.dataset.at[self._index, "cell_Y"] = props.centroid[1]
        self.dataset.at[self._index, "cell_shape_orientation_rad"] = props.cell_shape_orientation_rad
        self.dataset.at[self._index, "cell_shape_orientation_deg"] = props.cell_shape_orientation_deg
        self.dataset.at[self._index, "cell_major_axis_length"] = props.major_axis_length
        self.dataset.at[self._index, "cell_minor_axis_length"] = props.minor_axis_length
        self.dataset.at[self._index, "cell_eccentricity"] = props.eccentricity
        self.dataset.at[self._index, "cell_major_to_minor_ratio"] = props.cell_major_to_minor_ratio
        self.dataset.at[self._index, "cell_area"] = props.area
        self.dataset.at[self._index, "cell_perimeter"] = props.perimeter
        self.dataset.at[self._index, "cell_corner_points"] = json.dumps(props.cell_corner_points.tolist())

    def add_sc_organelle_props(self, organelle_props):
        """Fills the dataset with the single cell organelle properties."""
        self.dataset.at[self._index, "organelle_X"] = organelle_props.centroid[0]
        self.dataset.at[self._index, "organelle_Y"] = organelle_props.centroid[1]
        self.dataset.at[self._index, "nuc_organelle_distance"] = organelle_props.nuc_organelle_distance
        self.dataset.at[self._index, "organelle_orientation_rad"] = organelle_props.organelle_orientation_rad
        self.dataset.at[self._index, "organelle_orientation_deg"] = organelle_props.organelle_orientation_deg

    def add_sc_marker_nuclei_props(self, marker_nuc_props):
        """Fills the dataset with the single cell marker nuclei properties."""
        self.dataset.at[self._index, "marker_mean_expression_nuc"] = marker_nuc_props.mean_intensity
        self.dataset.at[self._index, "marker_sum_expression_nuc"] = marker_nuc_props.marker_sum_expression_nuc
        self.dataset.at[self._index, "marker_nucleus_orientation_rad"] = marker_nuc_props.marker_nucleus_orientation_rad
        self.dataset.at[self._index, "marker_nucleus_orientation_deg"] = marker_nuc_props.marker_nucleus_orientation_deg

    def add_sc_marker_nuclei_cytosol_props(self, marker_nuc_cyt_props):
        """Fills the dataset with the single cell marker nuclei cytosol properties."""
        self.dataset.at[self._index, "marker_mean_expression_cyt"] = marker_nuc_cyt_props.mean_intensity
        self.dataset.at[self._index, "marker_sum_expression_cyt"] = marker_nuc_cyt_props.marker_sum_expression_cyt

    def add_sc_marker_membrane_props(self, marker_membrane_props):
        """Fills the dataset with the single cell marker membrane properties."""
        self.dataset.at[self._index, "marker_mean_expression_mem"] = marker_membrane_props.mean_intensity
        self.dataset.at[self._index, "marker_sum_expression_mem"] = marker_membrane_props.marker_sum_expression_mem

    def add_sc_junction_props(self, sc_junction_props):
        """Fills the dataset with the single cell junction properties."""
        junctions_centroid_x, junctions_centroid_y = sc_junction_props.sc_junction_protein_circular_props.weighted_centroid

        self.dataset.at[self._index, "junction_centroid_X"] = junctions_centroid_x
        self.dataset.at[self._index, "junction_centroid_Y"] = junctions_centroid_y
        self.dataset.at[self._index, "junction_perimeter"] = sc_junction_props.interface_perimeter
        self.dataset.at[self._index, "junction_protein_area"] = sc_junction_props.sc_junction_protein_props.area
        # dataset.at[index, "junction_fragmented_perimeter"] = sc_junction_props.junction_fragmented_perimeter
        self.dataset.at[
            self._index, "junction_mean_expression"] = sc_junction_props.sc_junction_interface_props.mean_intensity
        self.dataset.at[
            self._index, "junction_protein_intensity"] = sc_junction_props.junction_protein_intensity

    def add_sc_junction_sec_stat_props(self, sc_junction_props):
        # dataset.at[index, "junction_coverage_index"] = sc_junction_props.junction_coverage_index
        self.dataset.at[self._index, "junction_interface_linearity_index"] = sc_junction_props.junction_interface_linearity_index
        self.dataset.at[self._index, "junction_interface_occupancy"] = sc_junction_props.junction_interface_occupancy
        self.dataset.at[
            self._index, "junction_intensity_per_interface_area"] = sc_junction_props.junction_intensity_per_interface_area
        self.dataset.at[self._index, "junction_cluster_density"] = sc_junction_props.junction_cluster_density

    def add_morans_i_props(self, morans_i):
        self.dataset.at[self._index, "morans_i"] = morans_i.I
        self.dataset.at[self._index, "morans_p_norm"] = morans_i.p_norm

    def add_neighborhood_props(self, neighborhood_props):
        self.dataset.at[self._index, "neighbors_cell"] = neighborhood_props.num_neighbours

        # fill properties for first nearest neighbors
        self.dataset.at[self._index, "neighbors_mean_dif_1st"] = neighborhood_props.mean_dif_first_neighbors
        self.dataset.at[self._index, "neighbors_median_dif_1st"] = neighborhood_props.median_dif_first_neighbors
        self.dataset.at[self._index, "neighbors_stddev_dif_1st"] = neighborhood_props.var_dif_first_neighbors
        self.dataset.at[self._index, "neighbors_range_dif_1st"] = neighborhood_props.range_dif_first_neighbors

        # fill properties for second-nearest neighbors
        self.dataset.at[self._index, "neighbors_mean_dif_2nd"] = neighborhood_props.mean_dif_second_neighbors
        self.dataset.at[self._index, "neighbors_median_dif_2nd"] = neighborhood_props.median_dif_second_neighbors
        self.dataset.at[self._index, "neighbors_stddev_dif_2nd"] = neighborhood_props.var_dif_second_neighbors
        self.dataset.at[self._index, "neighbors_range_dif_2nd"] = neighborhood_props.range_dif_second_neighbors
