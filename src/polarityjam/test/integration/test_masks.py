from unittest.mock import patch

from polarityjam import (
    BioMedicalInstanceSegmentation,
    BioMedicalInstanceSegmentationMask,
)
from polarityjam.test.test_common import TestCommon
from polarityjam.utils.io import read_image


class TestBioMedicalMask(TestCommon):
    """"""

    def setUp(self) -> None:
        super().setUp()


class TestBioMedicalInstanceSegmentationMask(TestCommon):
    """"""

    def setUp(self) -> None:
        super().setUp()


class TestBioMedicalInstanceSegmentation(TestCommon):
    """Tests the BioMedicalInstanceSegmentation class"""

    def setUp(self) -> None:
        super().setUp()

    @patch("polarityjam.model.masks.BioMedicalInstanceSegmentation.update_graphs")
    def test__init__no_connection_graph(self, update_mock):
        # prepare
        np_img = read_image(self.get_test_mask("mask.png"))
        inst_seg_mask = BioMedicalInstanceSegmentationMask(np_img)
        # call
        r = BioMedicalInstanceSegmentation(inst_seg_mask, connection_graph=False)

        # assert
        self.assertEqual(11, len(r.neighborhood_graph.nodes))
        self.assertEqual(11, len(r.neighborhood_graph_connected.nodes))
        self.assertEqual(11, len(r.segmentation_mask.get_labels()))
        self.assertEqual(11, len(r.segmentation_mask_connected.get_labels()))
        self.assertIsNone(r.segmentation_mask_nuclei)
        self.assertIsNone(r.segmentation_mask_nuclei)
        update_mock.assert_not_called()

    def test__init__connection_graph(self):
        # prepare
        np_img = read_image(self.get_test_mask("mask.png"))
        inst_seg_mask = BioMedicalInstanceSegmentationMask(np_img)
        # call
        r = BioMedicalInstanceSegmentation(inst_seg_mask, connection_graph=True)

        labels_connected = r.segmentation_mask_connected.get_labels()
        labels = r.segmentation_mask.get_labels()

        self.assertIn(127, labels)
        labels.remove(127)

        self.assertIn(85, labels)
        labels.remove(85)

        self.assertListEqual(
            labels, labels_connected
        )  # exactly two islands have been removed

        # assert
        self.assertEqual(11, len(r.neighborhood_graph.nodes))
        self.assertEqual(9, len(r.neighborhood_graph_connected.nodes))
        self.assertEqual(11, len(r.segmentation_mask.get_labels()))
        self.assertEqual(9, len(r.segmentation_mask_connected.get_labels()))
        self.assertIsNone(r.segmentation_mask_nuclei)
        self.assertIsNone(r.segmentation_mask_nuclei)

    def test_segmentation_mask_nuclei(self):
        # prepare
        np_img = read_image(self.get_test_mask("mask.png"))
        inst_seg_mask = BioMedicalInstanceSegmentationMask(np_img)

        np_img_nuc = read_image(self.get_test_mask("mask_nuclei.png"))
        inst_seg_nuc_mask = BioMedicalInstanceSegmentationMask(np_img_nuc)
        self.assertEqual(12, len(inst_seg_nuc_mask.get_labels()))

        # call
        r = BioMedicalInstanceSegmentation(inst_seg_mask, connection_graph=False)
        r.segmentation_mask_nuclei = inst_seg_nuc_mask

        # assert
        self.assertEqual(11, len(r.segmentation_mask.get_labels()))
        self.assertEqual(
            11, len(r.segmentation_mask_nuclei.get_labels())
        )  # same labels as mask

    def test_segmentation_mask_nuclei_with_connection(self):
        # prepare
        np_img = read_image(self.get_test_mask("mask.png"))
        inst_seg_mask = BioMedicalInstanceSegmentationMask(np_img)

        np_img_nuc = read_image(self.get_test_mask("mask_nuclei.png"))
        inst_seg_nuc_mask = BioMedicalInstanceSegmentationMask(np_img_nuc)
        self.assertEqual(12, len(inst_seg_nuc_mask.get_labels()))

        # call
        r = BioMedicalInstanceSegmentation(inst_seg_mask, connection_graph=True)
        r.segmentation_mask_nuclei = inst_seg_nuc_mask

        # assert
        self.assertEqual(9, len(r.segmentation_mask_connected.get_labels()))
        self.assertEqual(
            9, len(r.segmentation_mask_nuclei.get_labels())
        )  # same labels as mask

    def test_segmentation_mask_organelle(self):
        # prepare
        np_img = read_image(self.get_test_mask("mask.png"))
        inst_seg_mask = BioMedicalInstanceSegmentationMask(np_img)

        np_img_orga = read_image(self.get_test_mask("mask_organelle.png"))
        inst_seg_orga_mask = BioMedicalInstanceSegmentationMask(np_img_orga)
        self.assertEqual(12, len(inst_seg_orga_mask.get_labels()))

        # call
        r = BioMedicalInstanceSegmentation(inst_seg_mask, connection_graph=False)
        r.segmentation_mask_organelle = inst_seg_orga_mask

        # assert
        self.assertEqual(11, len(r.segmentation_mask.get_labels()))
        self.assertEqual(
            11, len(r.segmentation_mask_organelle.get_labels())
        )  # same labels as mask

    def test_segmentation_mask_organelle_with_connection(self):
        # prepare
        np_img = read_image(self.get_test_mask("mask.png"))
        inst_seg_mask = BioMedicalInstanceSegmentationMask(np_img)

        np_img_orga = read_image(self.get_test_mask("mask_organelle.png"))
        inst_seg_orga_mask = BioMedicalInstanceSegmentationMask(np_img_orga)
        self.assertEqual(12, len(inst_seg_orga_mask.get_labels()))

        # call
        r = BioMedicalInstanceSegmentation(inst_seg_mask, connection_graph=True)
        r.segmentation_mask_organelle = inst_seg_orga_mask

        # assert
        self.assertEqual(9, len(r.segmentation_mask_connected.get_labels()))
        self.assertEqual(
            9, len(r.segmentation_mask_organelle.get_labels())
        )  # same labels as mask

    @patch("polarityjam.model.masks.BioMedicalInstanceSegmentation.update_graphs")
    def test_remove_labels_no_connection(self, update_mock):
        # prepare
        np_img = read_image(self.get_test_mask("mask.png"))
        inst_seg_mask = BioMedicalInstanceSegmentationMask(np_img)
        np_img_orga = read_image(self.get_test_mask("mask_organelle.png"))
        np_img_nuc = read_image(self.get_test_mask("mask_nuclei.png"))

        r = BioMedicalInstanceSegmentation(inst_seg_mask, connection_graph=False)
        r.segmentation_mask_organelle = BioMedicalInstanceSegmentationMask(np_img_orga)
        r.segmentation_mask_nuclei = BioMedicalInstanceSegmentationMask(np_img_nuc)

        labels = r.segmentation_mask.get_labels()
        labels_connected = r.segmentation_mask_connected.get_labels()
        labels_nuc = r.segmentation_mask_nuclei.get_labels()
        labels_orga = r.segmentation_mask_organelle.get_labels()

        self.assertListEqual(labels, labels_connected)
        self.assertListEqual(labels_connected, labels_nuc)
        self.assertListEqual(labels_nuc, labels_orga)
        self.assertEqual(11, len(labels))
        self.assertEqual(11, len(labels_nuc))
        self.assertEqual(11, len(labels_orga))

        # call
        r.remove_instance_label(labels[-1])

        # assert
        self.assertEqual(10, len(r.segmentation_mask.get_labels()))
        self.assertEqual(10, len(r.neighborhood_graph.nodes))
        self.assertEqual(10, len(r.segmentation_mask_nuclei.get_labels()))
        self.assertEqual(10, len(r.segmentation_mask_organelle.get_labels()))
        update_mock.assert_not_called()

    def test_remove_labels_with_connection_no_islands(self):
        # prepare
        np_img = read_image(self.get_test_mask("mask.png"))
        inst_seg_mask = BioMedicalInstanceSegmentationMask(np_img)
        np_img_orga = read_image(self.get_test_mask("mask_organelle.png"))
        np_img_nuc = read_image(self.get_test_mask("mask_nuclei.png"))

        r = BioMedicalInstanceSegmentation(inst_seg_mask, connection_graph=True)
        r.segmentation_mask_organelle = BioMedicalInstanceSegmentationMask(np_img_orga)
        r.segmentation_mask_nuclei = BioMedicalInstanceSegmentationMask(np_img_nuc)

        self.assertEqual(11, len(r.segmentation_mask.get_labels()))
        self.assertEqual(11, len(r.neighborhood_graph.nodes))
        self.assertEqual(9, len(r.segmentation_mask_connected.get_labels()))
        self.assertEqual(9, len(r.segmentation_mask_organelle.get_labels()))
        self.assertEqual(9, len(r.segmentation_mask_nuclei.get_labels()))

        labels_connected = r.segmentation_mask_connected.get_labels()
        labels_nuc = r.segmentation_mask_nuclei.get_labels()
        labels_orga = r.segmentation_mask_organelle.get_labels()

        self.assertListEqual(labels_connected, labels_nuc)
        self.assertListEqual(labels_nuc, labels_orga)

        # call
        self.assertIn(64, labels_connected)
        islands = r.remove_instance_label(64)  # no islands in this case

        # assert
        self.assertListEqual([], islands)
        self.assertEqual(10, len(r.segmentation_mask.get_labels()))
        self.assertEqual(10, len(r.neighborhood_graph.nodes))
        self.assertEqual(8, len(r.segmentation_mask_connected.get_labels()))
        self.assertEqual(8, len(r.segmentation_mask_nuclei.get_labels()))
        self.assertEqual(8, len(r.segmentation_mask_organelle.get_labels()))

    def test_remove_labels_with_connection_with_islands(self):
        # prepare
        np_img = read_image(self.get_test_mask("mask.png"))
        inst_seg_mask = BioMedicalInstanceSegmentationMask(np_img)
        np_img_orga = read_image(self.get_test_mask("mask_organelle.png"))
        np_img_nuc = read_image(self.get_test_mask("mask_nuclei.png"))

        r = BioMedicalInstanceSegmentation(inst_seg_mask, connection_graph=True)
        r.segmentation_mask_organelle = BioMedicalInstanceSegmentationMask(np_img_orga)
        r.segmentation_mask_nuclei = BioMedicalInstanceSegmentationMask(np_img_nuc)

        self.assertEqual(11, len(r.segmentation_mask.get_labels()))
        self.assertEqual(11, len(r.neighborhood_graph.nodes))
        self.assertEqual(9, len(r.segmentation_mask_connected.get_labels()))
        self.assertEqual(9, len(r.segmentation_mask_organelle.get_labels()))
        self.assertEqual(9, len(r.segmentation_mask_nuclei.get_labels()))

        labels_connected = r.segmentation_mask_connected.get_labels()
        labels_nuc = r.segmentation_mask_nuclei.get_labels()
        labels_orga = r.segmentation_mask_organelle.get_labels()

        self.assertListEqual(labels_connected, labels_nuc)
        self.assertListEqual(labels_nuc, labels_orga)

        # call
        self.assertIn(191, labels_connected)
        islands = r.remove_instance_label(
            191
        )  # creates an islands - 212 - that should be removed

        # assert
        self.assertListEqual([212], islands)
        self.assertEqual(10, len(r.segmentation_mask.get_labels()))
        self.assertEqual(10, len(r.neighborhood_graph.nodes))
        self.assertEqual(7, len(r.segmentation_mask_connected.get_labels()))
        self.assertEqual(7, len(r.segmentation_mask_nuclei.get_labels()))
        self.assertEqual(7, len(r.segmentation_mask_organelle.get_labels()))
