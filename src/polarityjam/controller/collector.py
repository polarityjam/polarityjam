import numpy as np
from scipy import ndimage as ndi
from typing import List
from polarityjam import RuntimeParameter, ImageParameter
from polarityjam.compute.compute import channel_threshold_otsu
from polarityjam.model.collection import PropertiesCollection
from polarityjam.model.masks import SingleCellMasksCollection, MasksCollection
from polarityjam.model.moran import Moran
from polarityjam.model.properties import SingleCellCellProps, SingleCellNucleusProps, SingleCellOrganelleProps, \
    SingleCellMarkerProps, SingleCellMarkerMembraneProps, SingleCellMarkerNucleiProps, SingleCellMarkerCytosolProps, \
    SingleCellJunctionInterfaceProps, SingleCellJunctionProteinProps, SingleCellJunctionProteinCircularProps, \
    SingleCellJunctionProps, SingleCellPropertiesCollection, NeighborhoodProps


class PropertyCollector:
    """Static class that collects features "as they come" in a large dataset.
     Not responsible for feature calculation!
     """

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
    def add_out_path(props_collection: PropertiesCollection, filename: str, path: str):
        props_collection.out_path_dict[filename] = path

    @staticmethod
    def add_foi(props_collection: PropertiesCollection, filename: str, foi: str):
        props_collection.feature_of_interest_dict[filename] = foi

    @staticmethod
    def add_image_params(props_collection: PropertiesCollection, filename: str, image_params: ImageParameter):
        props_collection.image_parameter_dict[filename] = image_params

    @staticmethod
    def add_img(props_collection: PropertiesCollection, filename: str, img_nucleus: np.ndarray,
                img_junction: np.ndarray, img_marker: np.ndarray):
        # todo: check for duplication
        props_collection.img_channel_dict[filename] = {
            "nucleus": img_nucleus,
            "junction": img_junction,
            "marker": img_marker
        }

    @staticmethod
    def add_masks(props_collection: PropertiesCollection, filename: str, masks: MasksCollection):
        # todo: check for duplication
        props_collection.masks_dict[filename] = masks


class SingleCellPropertyCollector:

    def __init__(self, param: RuntimeParameter):
        self.param = param

    def calc_sc_props(self, sc_masks: SingleCellMasksCollection, im_marker: np.ndarray,
                      im_junction: np.ndarray) -> SingleCellPropertiesCollection:
        """calculates all properties for the single cell"""

        # properties for single cell
        sc_cell_props = self.calc_sc_cell_props(sc_masks.sc_mask.astype(int), self.param)

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
            sc_nuc_props = self.calc_sc_nucleus_props(sc_masks.sc_nucleus_mask.astype(int), sc_cell_props)

            # properties for organelle
            if sc_nuc_props and sc_masks.sc_organelle_mask is not None:
                sc_organelle_props = self.calc_sc_organelle_props(sc_masks.sc_organelle_mask.astype(int), sc_nuc_props)

        # properties for marker
        if im_marker is not None:
            sc_marker_props = self.calc_sc_marker_props(sc_masks.sc_mask.astype(int), im_marker)
            sc_marker_membrane_props = self.calc_sc_marker_membrane_props(sc_masks.sc_membrane_mask.astype(int),
                                                                          im_marker)

            # properties for marker nuclei
            if sc_masks.sc_nucleus_mask is not None:
                sc_marker_nuclei_props = self.calc_sc_marker_nuclei_props(sc_masks.sc_nucleus_mask.astype(int),
                                                                          im_marker, sc_nuc_props, sc_marker_props)
                sc_marker_cytosol_props = self.calc_sc_marker_cytosol_props(sc_masks.sc_cytosol_mask.astype(int),
                                                                            im_marker, sc_marker_nuclei_props)

        # properties for junctions
        if im_junction is not None:
            sc_junction_props = self.calc_sc_junction_props(
                sc_masks.sc_mask.astype(int),
                sc_masks.sc_membrane_mask.astype(int),
                sc_masks.sc_junction_protein_area_mask.astype(int),
                im_junction,
                sc_cell_props.minor_axis_length,
                self.param
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
    def calc_sc_cell_props(sc_mask: np.ndarray, param: RuntimeParameter) -> SingleCellCellProps:
        return SingleCellCellProps(sc_mask, param)

    @staticmethod
    def calc_sc_nucleus_props(sc_nucleus_maks: np.ndarray, sc_props: SingleCellCellProps) -> SingleCellNucleusProps:
        return SingleCellNucleusProps(sc_nucleus_maks, sc_props)

    @staticmethod
    def calc_sc_organelle_props(sc_organelle_mask: np.ndarray,
                                sc_nucleus_props: SingleCellNucleusProps) -> SingleCellOrganelleProps:
        return SingleCellOrganelleProps(sc_organelle_mask, sc_nucleus_props)

    @staticmethod
    def calc_sc_marker_props(sc_mask: np.ndarray, im_marker: np.ndarray) -> SingleCellMarkerProps:
        return SingleCellMarkerProps(sc_mask, im_marker)

    @staticmethod
    def calc_sc_marker_membrane_props(sc_membrane_mask: np.ndarray,
                                      im_marker: np.ndarray) -> SingleCellMarkerMembraneProps:
        return SingleCellMarkerMembraneProps(sc_membrane_mask, im_marker)

    @staticmethod
    def calc_sc_marker_nuclei_props(sc_nucleus_mask: np.ndarray, im_marker: np.ndarray,
                                    sc_nucleus_props: SingleCellNucleusProps,
                                    sc_marker_props: SingleCellMarkerProps) -> SingleCellMarkerNucleiProps:
        return SingleCellMarkerNucleiProps(sc_nucleus_mask, im_marker, sc_nucleus_props, sc_marker_props)

    @staticmethod
    def calc_sc_marker_cytosol_props(sc_cytosol_mask: np.ndarray, im_marker: np.ndarray,
                                     sc_marker_nuclei_props: SingleCellMarkerNucleiProps) -> SingleCellMarkerCytosolProps:
        return SingleCellMarkerCytosolProps(sc_cytosol_mask, im_marker, sc_marker_nuclei_props)

    @staticmethod
    def calc_sc_junction_props(sc_mask: np.ndarray, single_membrane_mask: np.ndarray,
                               single_junction_protein_area_mask: np.ndarray,
                               im_junction: np.ndarray, cell_minor_axis_length: float,
                               param: RuntimeParameter) -> SingleCellJunctionProps:

        im_junction_protein_single_cell = channel_threshold_otsu(im_junction, single_membrane_mask)

        sc_junction_interface_props = SingleCellJunctionInterfaceProps(single_membrane_mask, im_junction)

        sc_junction_protein_props = SingleCellJunctionProteinProps(single_junction_protein_area_mask,
                                                                   im_junction_protein_single_cell)

        sc_junction_protein_circular_props = SingleCellJunctionProteinCircularProps(
            im_junction_protein_single_cell,
            cell_minor_axis_length,
            sc_junction_interface_props.centroid
        )

        return SingleCellJunctionProps(sc_junction_interface_props, sc_junction_protein_props,
                                       sc_junction_protein_circular_props, sc_mask, param)


class SingleCellMaskCollector:

    @staticmethod
    def calc_sc_masks(masks: MasksCollection, connected_component_label: int, im_junction: np.ndarray,
                      membrane_thickness: int) -> SingleCellMasksCollection:
        sc_mask = SingleCellMaskCollector.get_sc_mask(masks.cell_mask_connected, connected_component_label)

        sc_membrane_mask = SingleCellMaskCollector.get_sc_membrane_mask(sc_mask, membrane_thickness)

        # init optional sc masks
        sc_nucleus_mask = None
        sc_organelle_mask = None
        sc_cytosol_mask = None
        sc_junction_protein_mask = None

        if masks.nuclei_mask is not None:
            sc_nucleus_mask = SingleCellMaskCollector.get_sc_nucleus_mask(masks.nuclei_mask, connected_component_label)
            sc_cytosol_mask = SingleCellMaskCollector.get_sc_cytosol_mask(sc_mask, sc_nucleus_mask)

        if masks.organelle_mask is not None:
            sc_organelle_mask = SingleCellMaskCollector.get_sc_organelle_mask(masks.organelle_mask,
                                                                              connected_component_label)

        if im_junction is not None:
            sc_junction_protein_mask = SingleCellMaskCollector.get_sc_junction_protein_mask(
                sc_membrane_mask, im_junction
            )

        return SingleCellMasksCollection(
            connected_component_label,
            sc_mask,
            sc_nucleus_mask,
            sc_organelle_mask,
            sc_membrane_mask,
            sc_cytosol_mask,
            sc_junction_protein_mask
        )

    @staticmethod
    def get_sc_mask(cell_mask_rem_island: np.ndarray, connected_component_label: int) -> np.ndarray:
        """Gets the single cell mask from a mask where each cell has an increasing connected component value."""
        sc_mask = np.where(cell_mask_rem_island == connected_component_label, 1, 0)
        return sc_mask

    @staticmethod
    def get_sc_nucleus_mask(nuclei_mask: np.ndarray, connected_component_label: int) -> np.ndarray:
        """Gets the single cell nucleus mask."""
        sc_nucleus_mask = np.where(nuclei_mask == connected_component_label, 1, 0)
        return sc_nucleus_mask

    @staticmethod
    def get_sc_organelle_mask(organelle_mask: np.ndarray, connected_component_label: int) -> np.ndarray:
        """Gets the single cell organelle mask."""
        sc_organelle_mask = np.where(organelle_mask == connected_component_label, 1, 0)

        return sc_organelle_mask

    @staticmethod
    def get_sc_membrane_mask(sc_mask: np.ndarray, membrane_thickness: int) -> np.ndarray:
        """Gets the single cell membrane mask."""
        sc_membrane_mask = SingleCellMaskCollector._get_outline_from_mask(sc_mask, membrane_thickness)
        return sc_membrane_mask

    @staticmethod
    def get_sc_cytosol_mask(sc_mask: np.ndarray, sc_nucleus_mask: np.ndarray) -> np.ndarray:
        """Gets the cytosol mask."""
        sc_cytosol_mask = np.logical_xor(sc_mask.astype(bool), sc_nucleus_mask.astype(bool))
        return sc_cytosol_mask

    @staticmethod
    def get_sc_junction_protein_mask(sc_membrane_mask: np.ndarray, im_junction: np.ndarray) -> np.ndarray:
        single_cell_junction_protein = channel_threshold_otsu(im_junction, sc_membrane_mask)
        return single_cell_junction_protein.astype(bool)

    @staticmethod
    def _get_outline_from_mask(mask: np.ndarray, width: int = 1) -> np.ndarray:
        """Computes outline for a mask with a single label"""
        mask = mask.astype(bool)
        dilated_mask = ndi.binary_dilation(mask, iterations=width)
        eroded_mask = ndi.binary_erosion(mask, iterations=width)
        outline_mask = np.logical_xor(dilated_mask, eroded_mask)

        return outline_mask
