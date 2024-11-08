"""Module for collecting features from the single cell images."""
from pathlib import Path
from typing import List, Union

from polarityjam import RuntimeParameter
from polarityjam.compute.neighborhood import k_neighbor_dif
from polarityjam.model.collection import PropertiesCollection
from polarityjam.model.image import BioMedicalImage, SingleCellImage
from polarityjam.model.masks import BioMedicalInstanceSegmentation
from polarityjam.model.moran import Moran, run_morans
from polarityjam.model.properties import (
    NeighborhoodProps,
    SingleCellPropertiesCollection,
)


class PropertyCollector:
    """Static class, collects features "as they come" in a large dataset.

    Not responsible for feature calculation!

    """

    @staticmethod
    def collect_sc_props(
        sc_prop_collection: SingleCellPropertiesCollection,
        props_collection: PropertiesCollection,
        filename: str,
        img_hash: str,
        connected_component_label: int,
        runtime_params: RuntimeParameter,
    ):
        """Collect single cell properties."""
        props_collection.add_sc_general_props(
            filename,
            img_hash,
            connected_component_label,
            sc_prop_collection.single_cell_props,
            runtime_params,
        )

        if sc_prop_collection.marker_props:
            props_collection.add_sc_marker_polarity_props(
                sc_prop_collection.marker_props, runtime_params
            )

        if sc_prop_collection.nucleus_props:
            props_collection.add_sc_nucleus_props(
                sc_prop_collection.nucleus_props, runtime_params
            )

        if sc_prop_collection.organelle_props:
            props_collection.add_sc_organelle_props(
                sc_prop_collection.organelle_props, runtime_params
            )

        if sc_prop_collection.marker_nuc_props:
            props_collection.add_sc_marker_nuclei_props(
                sc_prop_collection.marker_nuc_props, runtime_params
            )

        if (
            sc_prop_collection.marker_nuc_cyt_props
            and sc_prop_collection.marker_nuc_props
        ):
            props_collection.add_sc_marker_nuclei_cytosol_props(
                sc_prop_collection.marker_nuc_cyt_props,
                sc_prop_collection.marker_nuc_props,
                runtime_params,
            )

        if sc_prop_collection.marker_membrane_props:
            props_collection.add_sc_marker_membrane_props(
                sc_prop_collection.marker_membrane_props, runtime_params
            )

        if sc_prop_collection.junction_props:
            props_collection.add_sc_junction_props(
                sc_prop_collection.junction_props, runtime_params
            )

        props_collection.increase_index()

    @staticmethod
    def collect_group_statistic(
        props_collection: PropertiesCollection,
        morans_i: Moran,
        length: int,
        runtime_params: RuntimeParameter,
    ):
        """Collect group statistic."""
        props_collection.reset_index()
        for _ in range(1, length):
            props_collection.add_morans_i_props(morans_i, runtime_params)
            props_collection.increase_index()

    @staticmethod
    def collect_neighborhood_props(
        props_collection: PropertiesCollection,
        neighborhood_props_list: List[NeighborhoodProps],
        runtime_params: RuntimeParameter,
    ):
        """Collect neighborhood properties."""
        props_collection.reset_index()
        for neighborhood_props in neighborhood_props_list:
            props_collection.add_neighborhood_props(neighborhood_props, runtime_params)
            props_collection.increase_index()

    @staticmethod
    def get_foi(props_collection: PropertiesCollection, foi: str):
        """Get the Field of Interest (FOI) from the property collection."""
        return props_collection.dataset.at[props_collection.current_index() - 1, foi]

    @staticmethod
    def reset_index(props_collection: PropertiesCollection):
        """Reset the index of the property collection."""
        props_collection.reset_index()

    @staticmethod
    def set_reset_index(props_collection: PropertiesCollection):
        """Set the reset_index of the property collection."""
        props_collection.set_reset_index()

    @staticmethod
    def add_out_path(
        props_collection: PropertiesCollection, filename: str, path: Union[Path, str]
    ):
        """Add the output path to the property collection."""
        props_collection.out_path_dict[filename] = path

    @staticmethod
    def add_runtime_params(
        props_collection: PropertiesCollection,
        filename: str,
        runtime_params: RuntimeParameter,
    ):
        """Add the runtime parameters to the property collection."""
        props_collection.runtime_params_dict[filename] = runtime_params

    @staticmethod
    def add_img(
        props_collection: PropertiesCollection, filename: str, img: BioMedicalImage
    ):
        """Add the image to the property collection."""
        props_collection.img_dict[filename] = img

    @staticmethod
    def add_sc_imgs(
        props_collection: PropertiesCollection,
        filename: str,
        sc_img_list: List[SingleCellImage],
    ):
        """Add the single cell images to the property collection."""
        props_collection.sc_img_dict[filename] = sc_img_list


class GroupPropertyCollector:
    """Static class, collects group features "as they come" in a large dataset."""

    @staticmethod
    def calc_moran(
        bio_med_seg: BioMedicalInstanceSegmentation, feature_of_interest_name: str
    ):
        """Calculate Moran's I for a given feature of interest."""
        morans_i = run_morans(
            bio_med_seg.neighborhood_graph_connected, feature_of_interest_name
        )

        return morans_i

    @staticmethod
    def calc_neighborhood(
        bio_med_seg: BioMedicalInstanceSegmentation, feature_of_interest_name: str
    ):
        """Calculate neighborhood properties for a given feature of interest."""
        return k_neighbor_dif(
            bio_med_seg.neighborhood_graph_connected, feature_of_interest_name
        )


class SingleCellPropertyCollector:
    """Static class, responsible for collecting single cell properties."""

    @staticmethod
    def get_sc_props(
        sc_image: SingleCellImage,
        params: RuntimeParameter,
    ) -> SingleCellPropertiesCollection:
        """Calculate all properties for the single cell."""
        # init optional properties
        sc_nuc_props = None
        sc_organelle_props = None
        sc_marker_props = None
        sc_marker_membrane_props = None
        sc_marker_nuclei_props = None
        sc_marker_cytosol_props = None
        sc_junction_props = None

        sc_cell_props = sc_image.get_cell_properties(params)
        if sc_image.has_nuclei():
            sc_nuc_props = sc_image.get_nucleus_properties(params)

            if sc_image.has_organelle():
                sc_organelle_props = sc_image.get_organelle_properties(params)

        if sc_image.has_marker():
            sc_marker_props = sc_image.get_marker_properties(params)
            sc_marker_membrane_props = sc_image.get_marker_membrane_properties(params)

            if sc_image.has_nuclei():
                sc_marker_nuclei_props = sc_image.get_marker_nucleus_properties(params)
                sc_marker_cytosol_props = sc_image.get_marker_cytosol_properties(params)

        if sc_image.has_junction():
            sc_junction_props = sc_image.get_junction_properties(params)

        return SingleCellPropertiesCollection(
            sc_cell_props,
            sc_nuc_props,
            sc_organelle_props,
            sc_marker_props,
            sc_marker_membrane_props,
            sc_marker_nuclei_props,
            sc_marker_cytosol_props,
            sc_junction_props,
        )
