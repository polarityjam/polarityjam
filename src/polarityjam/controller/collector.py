from pathlib import Path
from typing import List, Union

import numpy as np

from polarityjam import RuntimeParameter
from polarityjam.compute.neighborhood import k_neighbor_dif
from polarityjam.model.collection import PropertiesCollection
from polarityjam.model.image import BioMedicalImage, BioMedicalChannel
from polarityjam.model.masks import BioMedicalInstanceSegmentation, \
    SingleCellMasksCollection, BioMedicalInstanceSegmentationMask, BioMedicalMask
from polarityjam.model.moran import Moran, run_morans
from polarityjam.model.properties import SingleCellProps, SingleCellNucleusProps, SingleCellOrganelleProps, \
    SingleCellMarkerProps, SingleCellMarkerMembraneProps, SingleCellMarkerNucleiProps, SingleCellMarkerCytosolProps, \
    SingleCellJunctionProps, SingleCellPropertiesCollection, NeighborhoodProps


class PropertyCollector:
    """Static class, collects features "as they come" in a large dataset. Not responsible for feature calculation!"""

    @staticmethod
    def collect_sc_props(
            sc_prop_collection: SingleCellPropertiesCollection,
            props_collection: PropertiesCollection,
            filename: str,
            img_hash: str,
            connected_component_label: int
    ):

        props_collection.add_sc_general_props(filename, img_hash, connected_component_label,
                                              sc_prop_collection.single_cell_props)

        if sc_prop_collection.marker_props:
            props_collection.add_sc_marker_polarity_props(sc_prop_collection.marker_props)

        if sc_prop_collection.nucleus_props:
            props_collection.add_sc_nucleus_props(sc_prop_collection.nucleus_props)

        if sc_prop_collection.organelle_props:
            props_collection.add_sc_organelle_props(sc_prop_collection.organelle_props)

        if sc_prop_collection.marker_nuc_props:
            props_collection.add_sc_marker_nuclei_props(sc_prop_collection.marker_nuc_props)

        if sc_prop_collection.marker_nuc_cyt_props:
            props_collection.add_sc_marker_nuclei_cytosol_props(sc_prop_collection.marker_nuc_cyt_props)

        if sc_prop_collection.marker_membrane_props:
            props_collection.add_sc_marker_membrane_props(sc_prop_collection.marker_membrane_props)

        if sc_prop_collection.junction_props:
            props_collection.add_sc_junction_props(sc_prop_collection.junction_props)

        props_collection.increase_index()

    @staticmethod
    def collect_group_statistic(props_collection: PropertiesCollection, morans_i: Moran, length: int):
        props_collection.reset_index()
        for i in range(1, length):
            props_collection.add_morans_i_props(morans_i)
            props_collection.increase_index()

    @staticmethod
    def collect_neighborhood_props(props_collection: PropertiesCollection,
                                   neighborhood_props_list: List[NeighborhoodProps]):
        props_collection.reset_index()
        for neighborhood_props in neighborhood_props_list:
            props_collection.add_neighborhood_props(neighborhood_props)
            props_collection.increase_index()

    @staticmethod
    def get_foi(props_collection: PropertiesCollection, foi: str):
        return props_collection.dataset.at[props_collection.current_index() - 1, foi]

    @staticmethod
    def reset_index(props_collection: PropertiesCollection):
        props_collection.reset_index()

    @staticmethod
    def set_reset_index(props_collection: PropertiesCollection):
        props_collection.set_reset_index()

    @staticmethod
    def add_out_path(props_collection: PropertiesCollection, filename: str, path: Union[Path, str]):
        props_collection.out_path_dict[filename] = path

    @staticmethod
    def add_runtime_params(props_collection: PropertiesCollection, filename: str, runtime_params: RuntimeParameter):
        props_collection.runtime_params_dict[filename] = runtime_params

    @staticmethod
    def add_img(props_collection: PropertiesCollection, filename: str, img: BioMedicalImage):
        props_collection.img_dict[filename] = img


class GroupPropertyCollector:
    """Static class, collects group features "as they come" in a large dataset."""

    @staticmethod
    def calc_moran(bio_med_seg: BioMedicalInstanceSegmentation, feature_of_interest_name: str):
        # morans I analysis based on FOI
        morans_i = run_morans(bio_med_seg.neighborhood_graph_connected, feature_of_interest_name)

        return morans_i

    @staticmethod
    def calc_neighborhood(bio_med_seg: BioMedicalInstanceSegmentation, feature_of_interest_name: str):
        return k_neighbor_dif(bio_med_seg.neighborhood_graph_connected, feature_of_interest_name)


class SingleCellPropertyCollector:
    """Static class, collects single cell features "as they come" in a large dataset."""

    @staticmethod
    def calc_sc_props(sc_masks: SingleCellMasksCollection,
                      img: BioMedicalImage, runtime_params: RuntimeParameter) -> SingleCellPropertiesCollection:
        """calculates all properties for the single cell"""

        # properties for single cell
        sc_cell_props = SingleCellPropertyCollector.calc_sc_cell_props(sc_masks.sc_mask, runtime_params)

        # init optional properties
        sc_nuc_props = None
        sc_organelle_props = None
        sc_marker_props = None
        sc_marker_membrane_props = None
        sc_marker_nuclei_props = None
        sc_marker_cytosol_props = None
        sc_junction_props = None

        # properties for nucleus:
        if sc_masks.sc_nucleus_mask is not None:
            sc_nuc_props = SingleCellPropertyCollector.calc_sc_nucleus_props(
                sc_masks.sc_nucleus_mask, sc_cell_props
            )

            # properties for organelle
            if sc_nuc_props and sc_masks.sc_organelle_mask is not None:
                sc_organelle_props = SingleCellPropertyCollector.calc_sc_organelle_props(
                    sc_masks.sc_organelle_mask, sc_nuc_props
                )

        # properties for marker
        if img.marker.data is not None:
            sc_marker_props = SingleCellPropertyCollector.calc_sc_marker_props(
                sc_masks.sc_mask, img.marker.data, runtime_params
            )
            sc_marker_membrane_props = SingleCellPropertyCollector.calc_sc_marker_membrane_props(
                sc_masks.sc_membrane_mask, img.marker.data
            )

            # properties for marker nuclei
            if sc_masks.sc_nucleus_mask is not None:
                sc_marker_nuclei_props = SingleCellPropertyCollector.calc_sc_marker_nuclei_props(
                    sc_masks.sc_nucleus_mask, img.marker.data, sc_nuc_props, sc_marker_props
                )
                sc_marker_cytosol_props = SingleCellPropertyCollector.calc_sc_marker_cytosol_props(
                    sc_masks.sc_cytosol_mask, img.marker, sc_marker_nuclei_props
                )

        if img.junction.data is not None:
            sc_junction_props = SingleCellPropertyCollector.calc_sc_junction_props(
                img.junction,
                sc_masks.sc_mask,
                sc_masks.sc_membrane_mask,
                sc_masks.sc_junction_protein_area_mask,
                runtime_params
            )

        return SingleCellPropertiesCollection(
            sc_cell_props,
            sc_nuc_props,
            sc_organelle_props,
            sc_marker_props,
            sc_marker_membrane_props,
            sc_marker_nuclei_props,
            sc_marker_cytosol_props,
            sc_junction_props
        )

    @staticmethod
    def calc_sc_cell_props(sc_mask: BioMedicalMask, param: RuntimeParameter) -> SingleCellProps:
        return SingleCellProps(sc_mask, param)

    @staticmethod
    def calc_sc_nucleus_props(sc_nucleus_maks: BioMedicalMask, sc_props: SingleCellProps) -> SingleCellNucleusProps:
        return SingleCellNucleusProps(sc_nucleus_maks, sc_props)

    @staticmethod
    def calc_sc_organelle_props(
            sc_organelle_mask: BioMedicalMask,
            sc_nucleus_props: SingleCellNucleusProps
    ) -> SingleCellOrganelleProps:
        return SingleCellOrganelleProps(sc_organelle_mask, sc_nucleus_props)

    @staticmethod
    def calc_sc_marker_props(
            sc_mask: BioMedicalMask, im_marker: BioMedicalChannel, runtime_parameter: RuntimeParameter
    ) -> SingleCellMarkerProps:
        return SingleCellMarkerProps(sc_mask, im_marker, runtime_parameter.cue_direction)

    @staticmethod
    def calc_sc_marker_membrane_props(
            sc_membrane_mask: BioMedicalMask,
            im_marker: BioMedicalChannel
    ) -> SingleCellMarkerMembraneProps:
        return SingleCellMarkerMembraneProps(sc_membrane_mask, im_marker)

    @staticmethod
    def calc_sc_marker_nuclei_props(
            sc_nucleus_mask: BioMedicalMask, im_marker: BioMedicalChannel,
            sc_nucleus_props: SingleCellNucleusProps,
            sc_marker_props: SingleCellMarkerProps
    ) -> SingleCellMarkerNucleiProps:
        return SingleCellMarkerNucleiProps(sc_nucleus_mask, im_marker, sc_nucleus_props, sc_marker_props)

    @staticmethod
    def calc_sc_marker_cytosol_props(
            sc_cytosol_mask: BioMedicalMask, im_marker: BioMedicalChannel,
            sc_marker_nuclei_props: SingleCellMarkerNucleiProps
    ) -> SingleCellMarkerCytosolProps:
        return SingleCellMarkerCytosolProps(sc_cytosol_mask, im_marker, sc_marker_nuclei_props)

    @staticmethod
    def calc_sc_junction_props(
            im_junction: BioMedicalChannel, sc_mask: BioMedicalMask,
            single_cell_membrane_mask: BioMedicalMask,
            single_cell_junction_intensity_mask: BioMedicalMask,
            runtime_parameter: RuntimeParameter
    ) -> SingleCellJunctionProps:

        return SingleCellJunctionProps(
            im_junction,
            single_cell_membrane_mask,
            single_cell_junction_intensity_mask,
            sc_mask,
            runtime_parameter.cue_direction,
            runtime_parameter.dp_epsilon
        )


class SingleCellMaskCollector:

    @staticmethod
    def calc_sc_masks(
            bio_med_img: BioMedicalImage,
            connected_component_label: int,
            membrane_thickness: int,
            nuclei_mask_seg: BioMedicalInstanceSegmentationMask,
            organelle_mask_seg: BioMedicalInstanceSegmentationMask
    ) -> SingleCellMasksCollection:
        sc_mask = bio_med_img.segmentation.segmentation_mask_connected.get_single_instance_maks(
            connected_component_label)
        sc_membrane_mask = sc_mask.get_outline_from_mask(membrane_thickness)

        # init optional sc masks
        sc_nucleus_mask = None
        sc_organelle_mask = None
        sc_cytosol_mask = None
        sc_junction_protein_mask = None

        if nuclei_mask_seg is not None:
            sc_nucleus_mask = nuclei_mask_seg.get_single_instance_maks(connected_component_label)
            sc_cytosol_mask = sc_nucleus_mask.operation(sc_mask, np.logical_xor)

        if organelle_mask_seg is not None:
            sc_organelle_mask = organelle_mask_seg.get_single_instance_maks(connected_component_label)

        if bio_med_img.junction is not None:
            masked_sc_junction_channel = bio_med_img.junction.mask(sc_membrane_mask)
            sc_junction_protein_mask = BioMedicalMask.from_threshold_otsu(
                masked_sc_junction_channel.data,
                gaussian_filter=None
            )

        return SingleCellMasksCollection(
            connected_component_label,
            sc_mask,
            sc_nucleus_mask,
            sc_organelle_mask,
            sc_membrane_mask,
            sc_cytosol_mask,
            sc_junction_protein_mask,
        )
