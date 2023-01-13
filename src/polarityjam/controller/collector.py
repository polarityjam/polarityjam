import numpy as np
from scipy import ndimage as ndi

from polarityjam import RuntimeParameter
from polarityjam.compute.compute import otsu_thresh_mask
from polarityjam.model.collection import PropertiesCollection
from polarityjam.model.masks import SingleCellMasksCollection
from polarityjam.model.properties import SingleCellCellProps, SingleCellNucleusProps, SingleCellOrganelleProps, \
    SingleCellMarkerProps, SingleCellMarkerMembraneProps, SingleCellMarkerNucleiProps, SingleCellMarkerCytosolProps, \
    SingleCellJunctionInterfaceProps, SingleCellJunctionProteinProps, SingleCellJunctionProteinCircularProps, \
    SingleCellJunctionProps, SingleCellPropertiesCollection


class PropertyCollector:
    """Collects features "as they come" in a large dataset. Not responsible for feature calculation!"""

    def __init__(self):
        pass

    def collect_sc_props(self, sc_prop_collection: SingleCellPropertiesCollection, props_collection: PropertiesCollection,
                         filename, img_hash, connected_component_label):

        props_collection.add_sc_general_props(filename, img_hash, connected_component_label, sc_prop_collection.single_cell_props)

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

        if sc_prop_collection.junction_props:
            props_collection.add_sc_junction_sec_stat_props(sc_prop_collection.junction_props)

        props_collection.increase_index()

    def collect_group_statistic(self, props_collection, morans_i, length):
        props_collection.reset_index()
        for i in range(1, length):
            props_collection.add_morans_i_props(morans_i)
            props_collection.increase_index()

    def collect_neighborhood_props(self, props_collection, neighborhood_props_list):
        props_collection.reset_index()
        for neighborhood_props in neighborhood_props_list:
            props_collection.add_neighborhood_props(neighborhood_props)
            props_collection.increase_index()

    def get_foi(self, props_collection, foi):
        return props_collection.dataset.at[props_collection.current_index() - 1, foi]

    def reset_index(self, props_collection):
        props_collection.reset_index()

    def set_reset_index(self, props_collection):
        props_collection.set_reset_index()

    def add_out_path(self, props_collection, filename, path):
        props_collection.out_path_dict[filename] = path

    def add_foi(self, props_collection, filename, foi):
        props_collection.feature_of_interest_dict[filename] = foi

    def add_img(self, props_collection, filename, img_nucleus, img_junction, img_marker):
        # todo: check for duplication
        props_collection.img_channel_dict[filename] = {
            "nucleus": img_nucleus,
            "junction": img_junction,
            "marker": img_marker
        }

    def add_masks(self, props_collection, filename, masks):
        # todo: check for duplication
        props_collection.masks_dict[filename] = masks


class SingleCellPropertyCollector:

    def __init__(self, param: RuntimeParameter):
        self.param = param

    def calc_sc_props(self, sc_masks, im_marker, im_junction):
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
    def calc_sc_cell_props(sc_mask, param: RuntimeParameter):
        return SingleCellCellProps(sc_mask, param)

    @staticmethod
    def calc_sc_nucleus_props(sc_nucleus_maks, sc_props: SingleCellCellProps):
        return SingleCellNucleusProps(sc_nucleus_maks, sc_props)

    @staticmethod
    def calc_sc_organelle_props(sc_organelle_mask, sc_nucleus_props: SingleCellNucleusProps):
        return SingleCellOrganelleProps(sc_organelle_mask, sc_nucleus_props)

    @staticmethod
    def calc_sc_marker_props(sc_mask, im_marker):
        return SingleCellMarkerProps(sc_mask, im_marker)

    @staticmethod
    def calc_sc_marker_membrane_props(sc_membrane_mask, im_marker):
        return SingleCellMarkerMembraneProps(sc_membrane_mask, im_marker)

    @staticmethod
    def calc_sc_marker_nuclei_props(sc_nucleus_mask, im_marker, sc_nucleus_props: SingleCellNucleusProps,
                                    sc_marker_props: SingleCellMarkerProps):
        return SingleCellMarkerNucleiProps(sc_nucleus_mask, im_marker, sc_nucleus_props, sc_marker_props)

    @staticmethod
    def calc_sc_marker_cytosol_props(sc_cytosol_mask, im_marker, sc_marker_nuclei_props: SingleCellMarkerNucleiProps):
        return SingleCellMarkerCytosolProps(sc_cytosol_mask, im_marker, sc_marker_nuclei_props)

    @staticmethod
    def calc_sc_junction_props(sc_mask, single_membrane_mask, single_junction_protein_area_mask,
                               im_junction, cell_minor_axis_length, param: RuntimeParameter):

        im_junction_protein_single_cell = otsu_thresh_mask(single_membrane_mask, im_junction)

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

    def __init__(self):
        pass

    def calc_sc_masks(self, masks, connected_component_label, im_junction, membrane_thickness):
        sc_mask = self.get_sc_mask(masks.cell_mask_rem_island, connected_component_label)

        sc_membrane_mask = self.get_sc_membrane_mask(sc_mask, membrane_thickness)

        # init optional sc masks
        sc_nucleus_mask = None
        sc_organelle_mask = None
        sc_cytosol_mask = None
        sc_junction_protein_mask = None

        if masks.nuclei_mask is not None:
            sc_nucleus_mask = self.get_sc_nucleus_mask(masks.nuclei_mask, connected_component_label)
            sc_cytosol_mask = self.get_sc_cytosol_mask(sc_mask, sc_nucleus_mask)

        if masks.organelle_mask is not None:
            sc_organelle_mask = self.get_sc_organelle_mask(masks.organelle_mask, connected_component_label)

        if im_junction is not None:
            sc_junction_protein_mask = self.get_sc_junction_protein_mask(
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
    def get_sc_mask(cell_mask_rem_island, connected_component_label):
        """Gets the single cell mask from a mask where each cell has an increasing connected component value."""
        sc_mask = np.where(cell_mask_rem_island == connected_component_label, 1, 0)
        return sc_mask

    @staticmethod
    def get_sc_nucleus_mask(nuclei_mask, connected_component_label):
        """Gets the single cell nucleus mask."""
        sc_nucleus_mask = np.where(nuclei_mask == connected_component_label, 1, 0)
        return sc_nucleus_mask

    @staticmethod
    def get_sc_organelle_mask(organelle_mask, connected_component_label):
        """Gets the single cell organelle mask."""
        sc_organelle_mask = np.where(organelle_mask == connected_component_label, 1, 0)

        return sc_organelle_mask

    @staticmethod
    def get_sc_membrane_mask(sc_mask, membrane_thickness):
        """Gets the single cell membrane mask."""
        sc_membrane_mask = SingleCellMaskCollector._get_outline_from_mask(sc_mask, membrane_thickness)
        return sc_membrane_mask

    @staticmethod
    def get_sc_cytosol_mask(sc_mask, sc_nucleus_mask):
        """Gets the cytosol mask."""
        sc_cytosol_mask = np.logical_xor(sc_mask.astype(bool), sc_nucleus_mask.astype(bool))
        return sc_cytosol_mask

    @staticmethod
    def get_sc_junction_protein_mask(sc_membrane_mask, im_junction):
        single_cell_junction_protein = otsu_thresh_mask(sc_membrane_mask, im_junction)
        return single_cell_junction_protein.astype(bool)

    @staticmethod
    def _get_outline_from_mask(mask, width=1):
        """Computes outline for a mask with a single label"""
        mask = mask.astype(bool)
        dilated_mask = ndi.binary_dilation(mask, iterations=width)
        eroded_mask = ndi.binary_erosion(mask, iterations=width)
        outline_mask = np.logical_xor(dilated_mask, eroded_mask)

        return outline_mask
