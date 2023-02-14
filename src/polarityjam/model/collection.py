import json

import pandas as pd

from polarityjam import RuntimeParameter
from polarityjam.model.image import BioMedicalImage
from polarityjam.model.moran import Moran
from polarityjam.model.properties import SingleCellMarkerProps, SingleCellNucleusProps, SingleCellProps, \
    SingleCellOrganelleProps, SingleCellMarkerNucleiProps, SingleCellMarkerCytosolProps, SingleCellMarkerMembraneProps, \
    SingleCellJunctionProps, NeighborhoodProps


class PropertiesCollection:
    """Collection of single cell properties"""

    def __init__(self):
        self.dataset = pd.DataFrame()
        self.out_path_dict = {}
        self.img_dict = {}
        self.runtime_params_dict = {}
        self._index = 1
        self._reset_index = 1

    def __len__(self):
        return len(self.dataset)

    def current_index(self):
        """Returns the current index of the dataset."""
        return self._index

    def increase_index(self):
        """Increases the index of the dataset."""
        self._index += 1

    def set_reset_index(self):
        """Set the reset index to the current index."""
        self._reset_index = self._index

    def reset_index(self):
        """Resets the index of the dataset."""
        self._index = self._reset_index

    def get_image_by_img_name(self, img_name: str) -> BioMedicalImage:
        """Get the image channel by image name.

        Args:
            img_name:
                image name of which the channel should be returned

        Returns:
            The image channel as numpy array

        """
        return self.img_dict[img_name]

    def get_runtime_params_by_img_name(self, img_name: str) -> RuntimeParameter:
        """Get the feature of interest given the image name.

        Args:
            img_name:
                The image name of which the feature of interest should be returned

        Returns:
            The feature of interest as numpy array

        """
        return self.runtime_params_dict[img_name]

    def get_properties_by_img_name(self, img_name: str, sorted: bool = True) -> pd.DataFrame:
        """Get the properties of the image given its image name.

        Args:
            sorted:
                If true, the properties are sorted by the label of the single cell
            img_name:
                The image name of which the properties should be returned

        Returns:
            The properties as a pandas dataset

        """
        features = self.dataset.loc[self.dataset["filename"] == img_name]
        if sorted:
            features = features.sort_values(by=["label"], ascending=True)
        return features

    def get_out_path_by_name(self, img_name: str) -> str:
        """Get the output path of the image given its image name.

        Args:
            img_name:
                The image name of which the output path should be returned

        Returns:
            The output path as string
        """
        return self.out_path_dict[img_name]

    def add_sc_marker_polarity_props(self, sc_marker_props: SingleCellMarkerProps):
        """Adds specific single cell marker props to the dataset.

        Args:
            sc_marker_props:
                The single cell marker properties

        """
        self.dataset.at[self._index, "marker_mean_expression"] = sc_marker_props.mean_intensity
        self.dataset.at[
            self._index, "marker_sum_expression"] = sc_marker_props.marker_sum_expression
        self.dataset.at[self._index, "marker_centroid_X"] = sc_marker_props.weighted_centroid[1]  # x is second index
        self.dataset.at[self._index, "marker_centroid_Y"] = sc_marker_props.weighted_centroid[0]
        self.dataset.at[
            self._index, "marker_centroid_orientation_rad"] = sc_marker_props.marker_centroid_orientation_rad
        self.dataset.at[
            self._index, "marker_centroid_orientation_deg"] = sc_marker_props.marker_centroid_orientation_deg

        self.dataset.at[
            self._index, "marker_cue_directional_intensity_ratio"] = sc_marker_props.marker_cue_directional_intensity_ratio
        self.dataset.at[
            self._index, "marker_cue_undirectional_intensity_ratio"] = sc_marker_props.marker_cue_undirectional_intensity_ratio

    def add_sc_nucleus_props(self, nucleus_props: SingleCellNucleusProps):
        """Adds specific single cell nucleus properties to the dataset.

        Args:
            nucleus_props:
                The single cell nucleus properties

        """
        self.dataset.at[self._index, "nuc_X"] = nucleus_props.centroid[1]  # x is second index
        self.dataset.at[self._index, "nuc_Y"] = nucleus_props.centroid[0]
        self.dataset.at[
            self._index, "nuc_displacement_orientation_rad"] = nucleus_props.nuc_displacement_orientation_rad
        self.dataset.at[
            self._index, "nuc_displacement_orientation_deg"] = nucleus_props.nuc_displacement_orientation_deg
        self.dataset.at[self._index, "nuc_shape_orientation_rad"] = nucleus_props.nuc_shape_orientation_rad
        self.dataset.at[self._index, "nuc_shape_orientation_deg"] = nucleus_props.nuc_shape_orientation_deg
        self.dataset.at[self._index, "nuc_major_axis_length"] = nucleus_props.major_axis_length
        self.dataset.at[self._index, "nuc_minor_axis_length"] = nucleus_props.minor_axis_length
        self.dataset.at[self._index, "nuc_area"] = nucleus_props.area
        self.dataset.at[self._index, "nuc_perimeter"] = nucleus_props.perimeter
        self.dataset.at[self._index, "nuc_eccentricity"] = nucleus_props.eccentricity
        self.dataset.at[self._index, "nuc_major_to_minor_ratio"] = nucleus_props.nuc_major_to_minor_ratio

    def add_sc_general_props(
            self, filename: str, img_hash: str, connected_component_label: int, sc_props: SingleCellProps):
        """Adds general single cell properties to the dataset, including filename, image hash,
        connected component label and general properties.

        Args:
            filename:
                The filename of the image
            img_hash:
                The hash of the image
            connected_component_label:
                The connected component label of the single cell
            sc_props:
                The single cell properties

        """
        self.dataset.at[self._index, "filename"] = filename
        self.dataset.at[self._index, "img_hash"] = img_hash
        self.dataset.at[self._index, "label"] = connected_component_label
        self.dataset.at[self._index, "cell_X"] = sc_props.centroid[1]  # x is second index
        self.dataset.at[self._index, "cell_Y"] = sc_props.centroid[0]
        self.dataset.at[self._index, "cell_shape_orientation_rad"] = sc_props.cell_shape_orientation_rad
        self.dataset.at[self._index, "cell_shape_orientation_deg"] = sc_props.cell_shape_orientation_deg
        self.dataset.at[self._index, "cell_major_axis_length"] = sc_props.major_axis_length
        self.dataset.at[self._index, "cell_minor_axis_length"] = sc_props.minor_axis_length
        self.dataset.at[self._index, "cell_eccentricity"] = sc_props.eccentricity
        self.dataset.at[self._index, "cell_major_to_minor_ratio"] = sc_props.cell_major_to_minor_ratio
        self.dataset.at[self._index, "cell_area"] = sc_props.area
        self.dataset.at[self._index, "cell_perimeter"] = sc_props.perimeter
        self.dataset.at[self._index, "cell_corner_points"] = json.dumps(sc_props.cell_corner_points.tolist())

    def add_sc_organelle_props(self, organelle_props: SingleCellOrganelleProps):
        """Adds specific single cell organelle properties to the dataset.

        Args:
            organelle_props:
                The single cell organelle properties

        """
        self.dataset.at[self._index, "organelle_X"] = organelle_props.centroid[1]  # x-axis is the second index
        self.dataset.at[self._index, "organelle_Y"] = organelle_props.centroid[0]
        self.dataset.at[self._index, "nuc_organelle_distance"] = organelle_props.nuc_organelle_distance
        self.dataset.at[self._index, "organelle_orientation_rad"] = organelle_props.organelle_orientation_rad
        self.dataset.at[self._index, "organelle_orientation_deg"] = organelle_props.organelle_orientation_deg

    def add_sc_marker_nuclei_props(self, marker_nuc_props: SingleCellMarkerNucleiProps):
        """Adds specific single cell marker-nuclei properties to the dataset.

        Args:
            marker_nuc_props:
                The single cell marker-nuclei properties

        """
        self.dataset.at[self._index, "marker_mean_expression_nuc"] = marker_nuc_props.mean_intensity
        self.dataset.at[self._index, "marker_sum_expression_nuc"] = marker_nuc_props.marker_sum_expression_nuc
        self.dataset.at[self._index, "marker_nucleus_orientation_rad"] = marker_nuc_props.marker_nucleus_orientation_rad
        self.dataset.at[self._index, "marker_nucleus_orientation_deg"] = marker_nuc_props.marker_nucleus_orientation_deg

    def add_sc_marker_nuclei_cytosol_props(self, marker_nuc_cyt_props: SingleCellMarkerCytosolProps):
        """Adds specific single cell marker-nuclei-cytosol properties to the dataset.

        Args:
            marker_nuc_cyt_props:
                The single cell marker-nuclei-cytosol properties

        """
        self.dataset.at[self._index, "marker_mean_expression_cyt"] = marker_nuc_cyt_props.mean_intensity
        self.dataset.at[self._index, "marker_sum_expression_cyt"] = marker_nuc_cyt_props.marker_sum_expression_cyt

    def add_sc_marker_membrane_props(self, marker_membrane_props: SingleCellMarkerMembraneProps):
        """Adds specific single cell marker-membrane properties to the dataset.

        Args:
            marker_membrane_props:
                The single cell marker-membrane properties

        """
        self.dataset.at[self._index, "marker_mean_expression_mem"] = marker_membrane_props.mean_intensity
        self.dataset.at[self._index, "marker_sum_expression_mem"] = marker_membrane_props.marker_sum_expression_mem

    def add_sc_junction_props(self, sc_junction_props: SingleCellJunctionProps):
        """Adds specific single cell junction properties to the dataset.

        Args:
            sc_junction_props:
                The single cell junction properties

        """
        j_centroid_first, j_centroid_second = sc_junction_props.sc_junction_intensity_props.weighted_centroid

        self.dataset.at[self._index, "junction_centroid_X"] = j_centroid_second  # x-axis is the second index
        self.dataset.at[self._index, "junction_centroid_Y"] = j_centroid_first
        self.dataset.at[self._index, "junction_perimeter"] = sc_junction_props.interface_perimeter
        self.dataset.at[self._index, "junction_protein_area"] = sc_junction_props.sc_junction_intensity_props.area
        # dataset.at[index, "junction_fragmented_perimeter"] = sc_junction_props.junction_fragmented_perimeter
        self.dataset.at[
            self._index, "junction_mean_expression"] = sc_junction_props.sc_junction_interface_props.mean_intensity
        self.dataset.at[
            self._index, "junction_protein_intensity"] = sc_junction_props.junction_protein_intensity
        self.dataset.at[
            self._index, "junction_interface_linearity_index"] = sc_junction_props.junction_interface_linearity_index
        self.dataset.at[self._index, "junction_interface_occupancy"] = sc_junction_props.junction_interface_occupancy
        self.dataset.at[
            self._index, "junction_intensity_per_interface_area"] = sc_junction_props.junction_intensity_per_interface_area
        self.dataset.at[self._index, "junction_cluster_density"] = sc_junction_props.junction_cluster_density
        self.dataset.at[
            self._index, "junction_cue_directional_intensity_ratio"] = sc_junction_props.junction_cue_directional_intensity_ratio
        self.dataset.at[
            self._index, "junction_cue_undirectional_intensity_ratio"] = sc_junction_props.junction_cue_undirectional_intensity_ratio

    def add_morans_i_props(self, morans_i: Moran):
        """Adds Moran's I value to the dataset.

        Args:
            morans_i:
                Moran's I object

        """
        self.dataset.at[self._index, "morans_i"] = morans_i.I
        self.dataset.at[self._index, "morans_p_norm"] = morans_i.p_norm

    def add_neighborhood_props(self, neighborhood_props: NeighborhoodProps):
        """Adds neighborhood properties to the dataset.

        Args:
            neighborhood_props:
                The neighborhood properties

        """
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
