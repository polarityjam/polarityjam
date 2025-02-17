import sys
from pathlib import Path

import pandas as pd

from polarityjam.argument_parsing import startup
from polarityjam.test.test_common import TestCommon


class TestExtractionOptions(TestCommon):
    def setUp(self) -> None:
        super().setUp()

    def test_extract_all_features(self):
        # create parameter file
        param_file = str(self.get_test_file("parameters_extract_all.yml"))
        in_file = str(self.get_test_file("060721_EGM2_18dyn_02_small.tif"))
        out_path = str(self.output_path.joinpath("run_small"))

        # build arguments
        sys.argv = [sys.argv[0]] + [
            "run",
            param_file,
            in_file,
            out_path,
            "--filename_prefix=myfile",
        ]

        # call
        startup()

        # assert
        df = pd.read_csv(Path(out_path).joinpath("myfile.csv"))
        self.assertEqual(df.shape[0], 2)  # 2 cells
        self.assertEqual(df.shape[1], 80)  # 79 features

    def test_no_polarity_extraction(self):
        # create parameter file
        param_file = str(self.get_test_file("parameters_extract_no_polarity.yml"))
        in_file = str(self.get_test_file("060721_EGM2_18dyn_02_small.tif"))
        out_path = str(self.output_path.joinpath("run_small"))

        # build arguments
        sys.argv = [sys.argv[0]] + [
            "run",
            param_file,
            in_file,
            out_path,
            "--filename_prefix=myfile",
        ]

        # call
        startup()

        # assert
        df = pd.read_csv(Path(out_path).joinpath("myfile.csv"))
        self.assertEqual(df.shape[0], 2)  # 2 cells
        self.assertEqual(df.shape[1], 63)  # 63 features (no polarity)

    def test_no_morphology_extraction(self):
        # create parameter file
        param_file = str(self.get_test_file("parameters_extract_no_morphology.yml"))
        in_file = str(self.get_test_file("060721_EGM2_18dyn_02_small.tif"))
        out_path = str(self.output_path.joinpath("run_small"))

        # build arguments
        sys.argv = [sys.argv[0]] + [
            "run",
            param_file,
            in_file,
            out_path,
            "--filename_prefix=myfile",
        ]

        # call
        startup()

        # assert
        df = pd.read_csv(Path(out_path).joinpath("myfile.csv"))
        self.assertEqual(df.shape[0], 2)  # 2 cells
        self.assertEqual(df.shape[1], 54)  # 54 features (no morphology)

    def test_no_intensity_extraction(self):
        # create parameter file
        param_file = str(self.get_test_file("parameters_extract_no_intensity.yml"))
        in_file = str(self.get_test_file("060721_EGM2_18dyn_02_small.tif"))
        out_path = str(self.output_path.joinpath("run_small"))

        # build arguments
        sys.argv = [sys.argv[0]] + [
            "run",
            param_file,
            in_file,
            out_path,
            "--filename_prefix=myfile",
        ]

        # call
        startup()

        # assert
        df = pd.read_csv(Path(out_path).joinpath("myfile.csv"))
        self.assertEqual(df.shape[0], 2)  # 2 cells
        self.assertEqual(df.shape[1], 69)  # 69 features (no intensity)

    def test_no_topology_extraction(self):
        # create parameter file
        param_file = str(self.get_test_file("parameters_extract_no_topology.yml"))
        in_file = str(self.get_test_file("060721_EGM2_18dyn_02_small.tif"))
        out_path = str(self.output_path.joinpath("run_small"))

        # build arguments
        sys.argv = [sys.argv[0]] + [
            "run",
            param_file,
            in_file,
            out_path,
            "--filename_prefix=myfile",
        ]

        # call
        startup()

        # assert
        df = pd.read_csv(Path(out_path).joinpath("myfile.csv"))
        self.assertEqual(df.shape[0], 2)  # 2 cells
        self.assertEqual(df.shape[1], 69)  # 69 features (no topology)
