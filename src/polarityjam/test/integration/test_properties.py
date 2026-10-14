import numpy as np
from skimage.draw import ellipse

from polarityjam import (
    BioMedicalImage,
    BioMedicalInstanceSegmentation,
    BioMedicalMask,
    Extractor,
    ImageParameter,
    PropertiesCollection,
    RuntimeParameter,
)
from polarityjam.compute.corner import get_contour
from polarityjam.compute.shape import (
    get_cue_side,
    mirror_along_cue_direction,
    partition_single_cell_mask,
)
from polarityjam.model.image import BioMedicalChannel, SingleCellImage
from polarityjam.model.properties import (
    SingleCellMarkerCytosolProps,
    SingleCellMarkerNucleiProps,
    SingleCellMarkerProps,
    SingleCellNucleusProps,
    SingleCellProps,
)
from polarityjam.test.test_common import TestCommon


class TestSingleCellProps(TestCommon):
    """"""

    def setUp(self) -> None:
        super().setUp()

    def setup_marker_img(self):
        cytosol = np.load(self.get_test_single_cell("cytosol.npy"))
        # directional positive and negative effect
        marker_de_p = np.load(self.get_test_single_cell("marker_de_p.npy"))
        marker_de_n = np.load(self.get_test_single_cell("marker_de_n.npy"))
        # unidirectional positive and negative effect
        marker_ue_p = np.load(self.get_test_single_cell("marker_ue_p.npy"))
        marker_ue_n = np.load(self.get_test_single_cell("marker_ue_n.npy"))
        # directional no effect
        marker_dne = np.load(self.get_test_single_cell("marker_dne.npy"))
        marker_une = np.load(self.get_test_single_cell("marker_une.npy"))
        # test numpy images
        self.sc_de_p = np.stack([cytosol, marker_de_p], axis=-1)
        self.sc_de_n = np.stack([cytosol, marker_de_n], axis=-1)
        self.sc_ue_p = np.stack([cytosol, marker_ue_p], axis=-1)
        self.sc_ue_n = np.stack([cytosol, marker_ue_n], axis=-1)
        self.sc_dne = np.stack([cytosol, marker_dne], axis=-1)
        self.sc_une = np.stack([cytosol, marker_une], axis=-1)
        # image parameter object
        self.img_param = ImageParameter()
        self.img_param.channel_junction = 0
        self.img_param.channel_nucleus = -1
        self.img_param.channel_organelle = -1
        self.img_param.channel_expression_marker = 1
        # label
        label = BioMedicalMask(
            np.load(self.get_test_single_cell("label.npy"))
        ).to_instance_mask(1)
        # segmentation
        self.seg = BioMedicalInstanceSegmentation(label, connection_graph=False)

    def setup_junction_img(self):
        # directional positive and negative effect
        junction_de_p = np.load(self.get_test_single_cell("junction_de_p.npy"))
        junction_de_n = np.load(self.get_test_single_cell("junction_de_n.npy"))
        # unidirectional positive and negative effect
        junction_ue_p = np.load(self.get_test_single_cell("junction_ue_p.npy"))
        junction_ue_n = np.load(self.get_test_single_cell("junction_ue_n.npy"))
        # directional no effect
        junction_dne = np.load(self.get_test_single_cell("junction_dne.npy"))
        junction_une = np.load(self.get_test_single_cell("junction_une.npy"))
        # test numpy images
        self.sc_de_p = np.expand_dims(junction_de_p, axis=-1)
        self.sc_de_n = np.expand_dims(junction_de_n, axis=-1)
        self.sc_ue_p = np.expand_dims(junction_ue_p, axis=-1)
        self.sc_ue_n = np.expand_dims(junction_ue_n, axis=-1)
        self.sc_dne = np.expand_dims(junction_dne, axis=-1)
        self.sc_une = np.expand_dims(junction_une, axis=-1)
        # image parameter object
        self.img_param = ImageParameter()
        self.img_param.channel_junction = 0
        self.img_param.channel_nucleus = -1
        self.img_param.channel_organelle = -1
        self.img_param.channel_expression_marker = -1
        # label
        label = BioMedicalMask(
            np.load(self.get_test_single_cell("label.npy"))
        ).to_instance_mask(1)
        # segmentation
        self.seg = BioMedicalInstanceSegmentation(label, connection_graph=False)

    def test_marker_property_cue_directional_intensity(self):
        # prepare
        self.setup_marker_img()

        img_sc_de_p = BioMedicalImage(
            self.sc_de_p, self.img_param, segmentation=self.seg
        )
        img_sc_de_n = BioMedicalImage(
            self.sc_de_n, self.img_param, segmentation=self.seg
        )
        img_sc_dne = BioMedicalImage(self.sc_dne, self.img_param, segmentation=self.seg)

        sc_mask = img_sc_de_p.get_single_cell_mask(1)
        sc_membrane_mask = img_sc_de_p.get_single_membrane_mask(1, 3)
        sc_junction_protein_mask = img_sc_de_p.get_single_junction_mask(1, 10)

        sc_mask_f = np.flip(sc_mask.data, axis=0)
        contour = get_contour(sc_mask_f.astype(int))

        sc_img_de_p = SingleCellImage(
            img_sc_de_p,
            contour=contour,
            connected_component_label=1,
            single_cell_mask=sc_mask,
            single_cell_membrane_mask=sc_membrane_mask,
            single_junction_mask=sc_junction_protein_mask,
        )

        sc_img_de_n = SingleCellImage(
            img_sc_de_n,
            contour=contour,
            connected_component_label=1,
            single_cell_mask=sc_mask,
            single_cell_membrane_mask=sc_membrane_mask,
            single_junction_mask=sc_junction_protein_mask,
        )

        sc_img_dne = SingleCellImage(
            img_sc_dne,
            contour=contour,
            connected_component_label=1,
            single_cell_mask=sc_mask,
            single_cell_membrane_mask=sc_membrane_mask,
            single_junction_mask=sc_junction_protein_mask,
        )

        runtime_param = RuntimeParameter()

        # call
        marker_props_de_p = sc_img_de_p.get_marker_properties(runtime_param)
        marker_props_de_n = sc_img_de_n.get_marker_properties(runtime_param)
        marker_props_dne = sc_img_dne.get_marker_properties(runtime_param)

        # assert
        self.assertGreater(
            marker_props_de_p.marker_cue_directional_intensity_ratio, 0.9
        )
        self.assertLess(marker_props_de_n.marker_cue_directional_intensity_ratio, -0.9)
        self.assertAlmostEqual(
            marker_props_dne.marker_cue_directional_intensity_ratio, 0, delta=0.1
        )

    def test_marker_property_cue_axial_intensity(self):
        # prepare
        self.setup_marker_img()

        img_sc_ue_p = BioMedicalImage(
            self.sc_ue_p, self.img_param, segmentation=self.seg
        )
        img_sc_ue_n = BioMedicalImage(
            self.sc_ue_n, self.img_param, segmentation=self.seg
        )
        img_sc_une = BioMedicalImage(self.sc_une, self.img_param, segmentation=self.seg)

        sc_mask = img_sc_ue_p.get_single_cell_mask(1)
        sc_membrane_mask = img_sc_ue_p.get_single_membrane_mask(1, 3)
        sc_junction_protein_mask = img_sc_ue_p.get_single_junction_mask(1, 10)

        sc_mask_f = np.flip(sc_mask.data, axis=0)
        contour = get_contour(sc_mask_f.astype(int))

        sc_img_ue_p = SingleCellImage(
            img_sc_ue_p,
            contour=contour,
            connected_component_label=1,
            single_cell_mask=sc_mask,
            single_cell_membrane_mask=sc_membrane_mask,
            single_junction_mask=sc_junction_protein_mask,
        )

        sc_img_ue_n = SingleCellImage(
            img_sc_ue_n,
            contour=contour,
            connected_component_label=1,
            single_cell_mask=sc_mask,
            single_cell_membrane_mask=sc_membrane_mask,
            single_junction_mask=sc_junction_protein_mask,
        )

        sc_img_une = SingleCellImage(
            img_sc_une,
            contour=contour,
            connected_component_label=1,
            single_cell_mask=sc_mask,
            single_cell_membrane_mask=sc_membrane_mask,
            single_junction_mask=sc_junction_protein_mask,
        )

        runtime_param = RuntimeParameter()

        # call
        marker_props_ue_p = sc_img_ue_p.get_marker_properties(runtime_param)
        marker_props_ue_n = sc_img_ue_n.get_marker_properties(runtime_param)
        marker_props_une = sc_img_une.get_marker_properties(runtime_param)

        # assert
        self.assertGreater(marker_props_ue_p.marker_cue_axial_intensity_ratio, 0.65)
        self.assertLess(marker_props_ue_n.marker_cue_axial_intensity_ratio, 0.35)
        self.assertAlmostEqual(
            marker_props_une.marker_cue_axial_intensity_ratio, 0.5, delta=0.05
        )

    def test_junction_property_cue_directional_intensity_ratio(self):
        # prepare
        self.setup_junction_img()

        img_sc_de_p = BioMedicalImage(
            self.sc_de_p, self.img_param, segmentation=self.seg
        )
        img_sc_de_n = BioMedicalImage(
            self.sc_de_n, self.img_param, segmentation=self.seg
        )
        img_sc_dne = BioMedicalImage(self.sc_dne, self.img_param, segmentation=self.seg)

        sc_mask = img_sc_de_p.get_single_cell_mask(1)
        sc_membrane_mask = img_sc_de_p.get_single_membrane_mask(1, 3)

        sc_mask_f = np.flip(sc_mask.data, axis=0)
        contour = get_contour(sc_mask_f.astype(int))

        sc_img_de_p = SingleCellImage(
            img_sc_de_p,
            contour=contour,
            connected_component_label=1,
            single_cell_mask=sc_mask,
            single_cell_membrane_mask=sc_membrane_mask,
            single_junction_mask=img_sc_de_p.get_single_junction_mask(1, 10),
        )

        sc_img_de_n = SingleCellImage(
            img_sc_de_n,
            contour=contour,
            connected_component_label=1,
            single_cell_mask=sc_mask,
            single_cell_membrane_mask=sc_membrane_mask,
            single_junction_mask=img_sc_de_n.get_single_junction_mask(1, 10),
        )

        sc_img_dne = SingleCellImage(
            img_sc_dne,
            contour=contour,
            connected_component_label=1,
            single_cell_mask=sc_mask,
            single_cell_membrane_mask=sc_membrane_mask,
            single_junction_mask=img_sc_dne.get_single_junction_mask(1, 10),
        )

        runtime_param = RuntimeParameter()

        # call
        junction_props_de_p = sc_img_de_p.get_junction_properties(runtime_param)
        junction_props_de_n = sc_img_de_n.get_junction_properties(runtime_param)
        junction_props_dne = sc_img_dne.get_junction_properties(runtime_param)

        # assert
        self.assertGreater(
            junction_props_de_p.junction_cue_directional_intensity_ratio, 0.9
        )
        self.assertLess(
            junction_props_de_n.junction_cue_directional_intensity_ratio, -0.9
        )
        self.assertAlmostEqual(
            junction_props_dne.junction_cue_directional_intensity_ratio, 0, delta=0.14
        )

    def test_junction_property_cue_axial_intensity_ratio(self):
        # prepare
        self.setup_junction_img()

        img_sc_ue_p = BioMedicalImage(
            self.sc_ue_p, self.img_param, segmentation=self.seg
        )
        img_sc_ue_n = BioMedicalImage(
            self.sc_ue_n, self.img_param, segmentation=self.seg
        )
        img_sc_une = BioMedicalImage(self.sc_une, self.img_param, segmentation=self.seg)

        sc_mask = img_sc_ue_p.get_single_cell_mask(1)
        sc_membrane_mask = img_sc_ue_p.get_single_membrane_mask(1, 3)

        sc_mask_f = np.flip(sc_mask.data, axis=0)
        contour = get_contour(sc_mask_f.astype(int))

        sc_img_ue_p = SingleCellImage(
            img_sc_ue_p,
            contour=contour,
            connected_component_label=1,
            single_cell_mask=sc_mask,
            single_cell_membrane_mask=sc_membrane_mask,
            single_junction_mask=img_sc_ue_p.get_single_junction_mask(1, 10),
        )

        sc_img_ue_n = SingleCellImage(
            img_sc_ue_n,
            contour=contour,
            connected_component_label=1,
            single_cell_mask=sc_mask,
            single_cell_membrane_mask=sc_membrane_mask,
            single_junction_mask=img_sc_ue_n.get_single_junction_mask(1, 10),
        )

        sc_img_une = SingleCellImage(
            img_sc_une,
            contour=contour,
            connected_component_label=1,
            single_cell_mask=sc_mask,
            single_cell_membrane_mask=sc_membrane_mask,
            single_junction_mask=img_sc_une.get_single_junction_mask(1, 10),
        )

        runtime_param = RuntimeParameter()

        # call
        junction_props_ue_p = sc_img_ue_p.get_junction_properties(runtime_param)
        junction_props_ue_n = sc_img_ue_n.get_junction_properties(runtime_param)
        junction_props_une = sc_img_une.get_junction_properties(runtime_param)

        # assert
        self.assertGreater(junction_props_ue_p.junction_cue_axial_intensity_ratio, 0.65)
        self.assertLess(junction_props_ue_n.junction_cue_axial_intensity_ratio, 0.35)
        self.assertAlmostEqual(
            junction_props_une.junction_cue_axial_intensity_ratio,
            0.5,
            delta=0.05,
        )

    def test_flow_asymmetry(self):
        def iou(m0, m180):
            return np.sum(np.logical_and(m0, m180)) / np.sum(np.logical_or(m0, m180))

        cone = np.load(self.get_test_asymmetry("sc_cone.npy"))
        r_round = np.load(self.get_test_asymmetry("sc_round.npy"))
        oval = np.load(self.get_test_asymmetry("sc_oval.npy"))

        cue_direction = 90

        m0_cone = mirror_along_cue_direction(cone, cue_direction)
        m180_cone = mirror_along_cue_direction(cone, cue_direction + 180)

        m0_round = mirror_along_cue_direction(r_round, cue_direction)
        m180_round = mirror_along_cue_direction(r_round, cue_direction + 180)

        m0_ovale = mirror_along_cue_direction(oval, cue_direction)
        m180_oval = mirror_along_cue_direction(oval, cue_direction + 180)

        iou_cone = iou(m0_cone, m180_cone)
        iou_round = iou(m0_round, m180_round)
        iou_oval = iou(m0_ovale, m180_oval)

        self.assertAlmostEqual(iou_cone, 0.62, delta=0.01)
        self.assertGreater(iou_round, 0.99)
        self.assertGreater(iou_oval, 0.97)

        # change cue direction to 90 degrees
        cue_direction = 0

        m0_cone = mirror_along_cue_direction(cone, cue_direction)
        m180_cone = mirror_along_cue_direction(cone, cue_direction + 180)

        m0_round = mirror_along_cue_direction(r_round, cue_direction)
        m180_round = mirror_along_cue_direction(r_round, cue_direction + 180)

        m0_ovale = mirror_along_cue_direction(oval, cue_direction)
        m180_oval = mirror_along_cue_direction(oval, cue_direction + 180)

        iou_cone = iou(m0_cone, m180_cone)
        iou_round = iou(m0_round, m180_round)
        iou_oval = iou(m0_ovale, m180_oval)

        self.assertAlmostEqual(iou_cone, 0.86, delta=0.01)
        self.assertGreater(iou_round, 0.98)
        self.assertGreater(iou_oval, 0.98)

        # change cue direction to 45 degrees
        cue_direction = 45

        m0_cone = mirror_along_cue_direction(cone, cue_direction)
        m180_cone = mirror_along_cue_direction(cone, cue_direction + 180)

        m0_round = mirror_along_cue_direction(r_round, cue_direction)
        m180_round = mirror_along_cue_direction(r_round, cue_direction + 180)

        m0_ovale = mirror_along_cue_direction(oval, cue_direction)
        m180_oval = mirror_along_cue_direction(oval, cue_direction + 180)

        iou_cone = iou(m0_cone, m180_cone)
        iou_round = iou(m0_round, m180_round)
        iou_oval = iou(m0_ovale, m180_oval)

        self.assertAlmostEqual(iou_cone, 0.37, delta=0.01)
        self.assertGreater(iou_round, 0.99)
        self.assertAlmostEqual(iou_oval, 0.37, delta=0.01)

    def test_nucleus_membrane_position(self):
        def ellipse_mask(center, radius_row, radius_col):
            mask = np.zeros((200, 300), dtype=bool)
            mask[ellipse(*center, radius_row, radius_col, shape=mask.shape)] = True
            return BioMedicalMask(mask)

        cell_mask = ellipse_mask((100, 150), 40, 100)
        sc_props = SingleCellProps(cell_mask, cell_mask, 0, 0)

        # call
        centered = SingleCellNucleusProps(ellipse_mask((100, 150), 15, 15), sc_props)
        off_center = SingleCellNucleusProps(ellipse_mask((85, 190), 15, 15), sc_props)
        touching = SingleCellNucleusProps(ellipse_mask((75, 150), 15, 15), sc_props)
        # long nucleus axis along the short cell axis
        rotated = SingleCellNucleusProps(ellipse_mask((100, 150), 25, 12), sc_props)

        # assert
        self.assertEqual(centered.nuc_membrane_distance, 25)
        self.assertAlmostEqual(centered.nuc_membrane_position_index, 1)
        self.assertAlmostEqual(off_center.nuc_membrane_position_index, 0.24)
        self.assertEqual(touching.nuc_membrane_distance, 0)
        self.assertEqual(touching.nuc_membrane_position_index, 0)
        self.assertEqual(rotated.nuc_membrane_distance, 15)
        self.assertAlmostEqual(rotated.nuc_membrane_position_index, 1)

    @staticmethod
    def setup_radial_marker(radius_row, radius_col, nucleus_radius=15):
        """Cell with a centered nucleus and its position on the rays from nucleus to membrane."""
        shape = (200, 300)
        rows, columns = np.indices(shape)

        cell_mask = np.zeros(shape, dtype=bool)
        cell_mask[ellipse(100, 150, radius_row, radius_col, shape=shape)] = True
        nucleus_mask = np.zeros(shape, dtype=bool)
        nucleus_mask[
            ellipse(100, 150, nucleus_radius, nucleus_radius, shape=shape)
        ] = True

        # 0 at the nucleus, 1 at the inner border of the membrane of thickness 5
        angle = np.arctan2(rows - 100, columns - 150)
        membrane_radius = (
            1 / np.hypot(np.cos(angle) / radius_col, np.sin(angle) / radius_row) - 5
        )
        position = (np.hypot(rows - 100, columns - 150) - nucleus_radius) / (
            membrane_radius - nucleus_radius
        )

        return cell_mask, nucleus_mask, np.clip(position, 0, 1)

    @staticmethod
    def get_marker_cytosol_props(
        cell_mask, nucleus_mask, intensity, cue_direction=0, num_radial_rings=None
    ):
        if num_radial_rings is None:
            num_radial_rings = RuntimeParameter().num_radial_rings

        cell_mask = BioMedicalMask(cell_mask)
        nucleus_mask = BioMedicalMask(nucleus_mask)
        marker = BioMedicalChannel(intensity)

        sc_props = SingleCellProps(cell_mask, cell_mask, 0, cue_direction)
        marker_nucleus_props = SingleCellMarkerNucleiProps(
            nucleus_mask,
            marker,
            SingleCellNucleusProps(nucleus_mask, sc_props),
            SingleCellMarkerProps(cell_mask, marker, [], []),
        )

        return SingleCellMarkerCytosolProps(
            nucleus_mask.operation(cell_mask, np.logical_xor),
            marker,
            marker_nucleus_props,
            cell_mask,
            cell_mask.get_outline_from_mask(5),
            cue_direction,
            num_radial_rings,
        )

    def test_marker_radial_gradient(self):
        round_cell, round_nucleus, round_position = self.setup_radial_marker(60, 60)
        oval_cell, oval_nucleus, oval_position = self.setup_radial_marker(40, 100)

        # intensity 10 at the nucleus, 30 at the membrane, bright nucleus
        round_intensity = np.where(round_nucleus, 100, 10 + 20 * round_position)
        oval_intensity = np.where(oval_nucleus, 100, 10 + 20 * oval_position)

        # bright membrane
        round_membrane = BioMedicalMask(round_cell).get_outline_from_mask(5).data
        round_intensity_membrane = np.where(round_membrane, 500, round_intensity)

        # call
        round_gradient = self.get_marker_cytosol_props(
            round_cell, round_nucleus, round_intensity
        )
        round_gradient_membrane = self.get_marker_cytosol_props(
            round_cell, round_nucleus, round_intensity_membrane
        )
        oval_gradient = self.get_marker_cytosol_props(
            oval_cell, oval_nucleus, oval_intensity
        )
        flat = self.get_marker_cytosol_props(
            oval_cell, oval_nucleus, np.full(oval_cell.shape, 20.0)
        )

        # assert
        self.assertAlmostEqual(round_gradient.marker_radial_gradient, 20, delta=0.5)
        self.assertAlmostEqual(
            round_gradient.marker_radial_gradient_norm, 1, delta=0.02
        )
        self.assertGreater(round_gradient.marker_radial_gradient_r2, 0.99)
        # the way from the nucleus to the membrane is 60 - 5 - 15 = 40 pixels long
        self.assertAlmostEqual(
            round_gradient.marker_radial_gradient_px, 0.5, delta=0.02
        )
        # (10 + 20 * 5 / 6) / (10 + 20 * 1 / 6)
        self.assertAlmostEqual(
            round_gradient.marker_radial_intensity_ratio, 2, delta=0.05
        )

        # the membrane is excluded
        self.assertEqual(
            round_gradient_membrane.marker_radial_gradient,
            round_gradient.marker_radial_gradient,
        )
        self.assertEqual(
            round_gradient_membrane.marker_radial_intensity_ratio,
            round_gradient.marker_radial_intensity_ratio,
        )

        # the normalized gradient is independent of the cell shape, the gradient per pixel is not
        self.assertAlmostEqual(
            oval_gradient.marker_radial_gradient_norm,
            round_gradient.marker_radial_gradient_norm,
            delta=0.1,
        )
        self.assertLess(
            oval_gradient.marker_radial_gradient_px,
            round_gradient.marker_radial_gradient_px / 2,
        )

        self.assertAlmostEqual(flat.marker_radial_gradient, 0)
        self.assertAlmostEqual(flat.marker_radial_gradient_norm, 0)
        self.assertTrue(np.isnan(flat.marker_radial_gradient_r2))
        self.assertAlmostEqual(flat.marker_radial_gradient_px, 0)
        self.assertAlmostEqual(flat.marker_radial_intensity_ratio, 1)

    def test_marker_radial_gradient_cue(self):
        cell_mask, nucleus_mask, position = self.setup_radial_marker(40, 100)
        rows, columns = np.indices(cell_mask.shape)

        # gradient only right of or above the nucleus, flat elsewhere
        right_gradient = np.where(columns > 150, 10 + 20 * position, 20)
        upper_gradient = np.where(rows < 100, 10 + 20 * position, 20)

        # call
        cue_right = self.get_marker_cytosol_props(
            cell_mask, nucleus_mask, right_gradient, cue_direction=0
        )
        cue_left = self.get_marker_cytosol_props(
            cell_mask, nucleus_mask, right_gradient, cue_direction=180
        )
        cue_up = self.get_marker_cytosol_props(
            cell_mask, nucleus_mask, upper_gradient, cue_direction=90
        )

        # assert
        self.assertAlmostEqual(
            cue_right.marker_radial_gradient_norm_cue_front, 1, delta=0.05
        )
        self.assertAlmostEqual(cue_right.marker_radial_gradient_norm_cue_back, 0)
        self.assertAlmostEqual(cue_left.marker_radial_gradient_norm_cue_front, 0)
        self.assertAlmostEqual(
            cue_left.marker_radial_gradient_norm_cue_back, 1, delta=0.05
        )
        self.assertAlmostEqual(
            cue_up.marker_radial_gradient_norm_cue_front, 1, delta=0.05
        )
        self.assertAlmostEqual(cue_up.marker_radial_gradient_norm_cue_back, 0)

    def test_marker_radial_gradient_no_cytosol(self):
        # the nucleus fills the cell up to the membrane
        cell_mask, nucleus_mask, _ = self.setup_radial_marker(30, 30, nucleus_radius=27)

        # call
        props = self.get_marker_cytosol_props(
            cell_mask, nucleus_mask, np.full(cell_mask.shape, 20.0)
        )

        # assert
        self.assertTrue(np.isnan(props.marker_radial_gradient))
        self.assertTrue(np.isnan(props.marker_radial_gradient_norm))
        self.assertTrue(np.isnan(props.marker_radial_gradient_px))
        self.assertTrue(np.isnan(props.marker_radial_intensity_ratio))
        self.assertTrue(np.isnan(props.marker_radial_gradient_norm_cue_front))

    def test_marker_radial_gradient_num_rings(self):
        cell_mask, nucleus_mask, position = self.setup_radial_marker(60, 60)
        intensity = np.where(nucleus_mask, 100, 10 + 20 * position)

        # call
        default_rings = self.get_marker_cytosol_props(
            cell_mask, nucleus_mask, intensity
        )
        few_rings = self.get_marker_cytosol_props(
            cell_mask, nucleus_mask, intensity, num_radial_rings=4
        )
        many_rings = self.get_marker_cytosol_props(
            cell_mask, nucleus_mask, intensity, num_radial_rings=20
        )

        # assert
        self.assertEqual(RuntimeParameter().num_radial_rings, 10)
        self.assertEqual(RuntimeParameter({"num_radial_rings": 4}).num_radial_rings, 4)
        self.assertEqual(len(default_rings._radial_profile()), 10)
        self.assertEqual(len(few_rings._radial_profile()), 4)
        self.assertEqual(len(many_rings._radial_profile()), 20)

        # a linear gradient does not depend on the number of rings
        self.assertAlmostEqual(
            few_rings.marker_radial_gradient_norm,
            default_rings.marker_radial_gradient_norm,
            delta=0.02,
        )
        self.assertAlmostEqual(
            many_rings.marker_radial_gradient_norm,
            default_rings.marker_radial_gradient_norm,
            delta=0.02,
        )

        with self.assertRaises(ValueError):
            self.get_marker_cytosol_props(
                cell_mask, nucleus_mask, intensity, num_radial_rings=1
            )

    def test_cue_side_matches_half_masks(self):
        cell_mask, _, _ = self.setup_radial_marker(40, 100)

        for cue_direction in [0, 45, 90, 180, 270]:
            # call
            half_masks, _, _ = partition_single_cell_mask(
                cell_mask, cue_direction, 300, 2
            )
            front = get_cue_side(cell_mask, (100, 150), cue_direction) > 0

            # assert
            # the first half mask lies in cue direction, differences only on the divisor line
            agreement = np.mean(front == half_masks[0][cell_mask].astype(bool))
            self.assertGreater(agreement, 0.98)

    def extract_radial_marker_img(self, marker_file, cue_direction=0):
        """Extract the features of the single cell with the given marker image."""
        junction = np.load(self.get_test_single_cell("cytosol.npy"))
        nucleus = np.load(self.get_test_single_cell("nucleus.npy"))
        marker = np.load(self.get_test_single_cell(marker_file))
        label = np.load(self.get_test_single_cell("label.npy")) > 127

        img_param = ImageParameter()
        img_param.channel_junction = 0
        img_param.channel_nucleus = 1
        img_param.channel_organelle = -1
        img_param.channel_expression_marker = 2

        runtime_param = RuntimeParameter()
        runtime_param.connection_graph = False  # a single cell has no neighbours
        runtime_param.cue_direction = cue_direction

        collection = PropertiesCollection()
        Extractor(runtime_param).extract(
            np.stack([junction, nucleus, marker], axis=-1),
            img_param,
            label.astype(np.uint8),
            marker_file,
            self.output_path,
            collection,
            segmentation_mask_nuclei=(nucleus > 127).astype(np.uint8),
        )

        return collection.dataset.iloc[0]

    def test_marker_radial_gradient_img(self):
        # call
        # radial positive and negative effect, intensity triples from nucleus to membrane or vice versa
        re_p = self.extract_radial_marker_img("marker_re_p.npy")
        re_n = self.extract_radial_marker_img("marker_re_n.npy")
        # radial no effect, but bright junctions
        rne = self.extract_radial_marker_img("marker_rne.npy")
        # radial positive effect only right of the nucleus, i.e. in front of it for cue direction 0
        rde_p = self.extract_radial_marker_img("marker_rde_p.npy")
        rde_p_reverse_cue = self.extract_radial_marker_img(
            "marker_rde_p.npy", cue_direction=180
        )

        # assert
        self.assertAlmostEqual(re_p["marker_radial_gradient_norm"], 1, delta=0.05)
        self.assertGreater(re_p["marker_radial_gradient_r2"], 0.99)
        self.assertGreater(re_p["marker_radial_gradient_px"], 0)
        self.assertAlmostEqual(re_p["marker_radial_intensity_ratio"], 2, delta=0.1)

        self.assertAlmostEqual(re_n["marker_radial_gradient_norm"], -1, delta=0.05)
        self.assertGreater(re_n["marker_radial_gradient_r2"], 0.99)
        self.assertLess(re_n["marker_radial_gradient_px"], 0)
        self.assertAlmostEqual(re_n["marker_radial_intensity_ratio"], 0.5, delta=0.05)

        # the junctions are part of the membrane, which is excluded
        self.assertAlmostEqual(rne["marker_radial_gradient_norm"], 0, delta=0.01)
        self.assertAlmostEqual(rne["marker_radial_intensity_ratio"], 1, delta=0.01)

        self.assertAlmostEqual(rde_p["marker_radial_gradient_norm"], 0.5, delta=0.05)
        self.assertAlmostEqual(
            rde_p["marker_radial_gradient_norm_cue_front"], 1, delta=0.05
        )
        self.assertAlmostEqual(
            rde_p["marker_radial_gradient_norm_cue_back"], 0, delta=0.01
        )
        self.assertAlmostEqual(
            rde_p_reverse_cue["marker_radial_gradient_norm_cue_front"], 0, delta=0.01
        )
        self.assertAlmostEqual(
            rde_p_reverse_cue["marker_radial_gradient_norm_cue_back"], 1, delta=0.05
        )
