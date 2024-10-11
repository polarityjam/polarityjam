import numpy as np

from polarityjam import (
    BioMedicalImage,
    BioMedicalInstanceSegmentation,
    BioMedicalMask,
    ImageParameter,
    RuntimeParameter,
)
from polarityjam.compute.corner import get_contour
from polarityjam.compute.shape import mirror_along_cue_direction
from polarityjam.model.image import SingleCellImage
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
