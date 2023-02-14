import glob
import platform
import sys
import unittest
from pathlib import Path

import pandas as pd

from polarityjam.argument_parsing import startup
from polarityjam.test.test_common import TestCommon
from polarityjam.utils.io import get_doc_file_prefix


class TestIntegration(TestCommon):

    def setUp(self) -> None:
        super().setUp()

    def check_doc_output(self, outputpath):
        outputpath = Path(outputpath)
        prefix = get_doc_file_prefix()

        log_file = outputpath.joinpath("%s%s" % (prefix, ".log"))
        param_file = outputpath.joinpath("%s%s" % (prefix, "_param.yml"))

        self.assertTrue(log_file.exists() and log_file.stat().st_size > 0, "Something is wrong with logging")
        self.assertTrue(
            param_file.exists() and param_file.stat().st_size > 0, "Something is wrong with copying the param file"
        )

    def test_run(self):
        # prepare
        in_file = str(self.get_test_image_path("060721_EGM2_18dyn_01.tif"))
        param_file = str(self.get_test_parameter_file("parameters_golgi_nuclei.yml"))
        out_path = str(self.output_path.joinpath("run"))

        # build arguments
        sys.argv = [sys.argv[0]] + ['run', param_file, in_file, out_path, '--filename_prefix=myfile']

        # call
        startup()

        # assert
        self.check_doc_output(out_path)

        # if version of cellpose is different, maybe labels change. Probably overall result not deterministic.
        # But staying in an interval is necessary. Hence, check number of segmented cells with error range of +-10%
        df = pd.read_csv(Path(out_path).joinpath("myfile.csv"))
        self.assertAlmostEqual(df.shape[0], 97, delta=10)

        # number of features should not change
        self.assertEqual(70, df.shape[1])

        # only one csv file in output
        num_csv = len(glob.glob(str(Path(out_path).joinpath("*.csv"))))
        self.assertEqual(1, num_csv)

    @unittest.skipIf(platform.system().lower() == 'windows', "Plotting too memory extensive. Skipping test!")
    def test_run_stack(self):
        in_path = str(self.get_test_image_folder("gn").joinpath("set_2"))
        param_file = str(self.get_test_parameter_file("parameters_golgi_nuclei.yml"))
        out_path = str(self.output_path.joinpath("run_stack"))

        # build arguments
        sys.argv = [sys.argv[0]] + ['run-stack', param_file, in_path, out_path]

        # call
        startup()

        # assert
        self.check_doc_output(out_path)

        df1 = pd.read_csv(Path(out_path).joinpath("060721_EGM2_18dyn_02.csv"))
        df2 = pd.read_csv(Path(out_path).joinpath("060721_EGM2_18dyn_03.csv"))

        self.assertAlmostEqual(107, df1.shape[0], delta=10)
        self.assertAlmostEqual(99, df2.shape[0], delta=10)

        # number of features should not change
        self.assertEqual(70, df1.shape[1])
        self.assertEqual(70, df2.shape[1])

        # two csv file in output
        num_csv = len(glob.glob(str(Path(out_path).joinpath("*.csv"))))
        self.assertEqual(3, num_csv)

    @unittest.skipIf(platform.system().lower() == 'windows', "Plotting too memory extensive. Skipping test!")
    def test_run_stack_no_golgi(self):
        in_path = str(self.get_test_image_folder("g"))
        param_file = str(self.get_test_parameter_file("parameters_no_golgi.yml"))
        out_path = str(self.output_path.joinpath("run_stack_no_golgi"))

        # build arguments
        sys.argv = [sys.argv[0]] + ['run-stack', param_file, in_path, out_path]

        # call
        startup()

        # assert
        self.check_doc_output(out_path)

        df1 = pd.read_csv(Path(out_path).joinpath("MAX_Flow_IF2_Vecad_Dapi_40X_I2.1.csv"))
        df2 = pd.read_csv(Path(out_path).joinpath("MAX_Flow_IF2_Vecad_Dapi_40X_I2.2.csv"))
        df3 = pd.read_csv(Path(out_path).joinpath("MAX_Flow_IF2_Vecad_Dapi_40X_I3.1.csv"))

        self.assertAlmostEqual(56, df1.shape[0], delta=5)
        self.assertAlmostEqual(56, df2.shape[0], delta=5)
        self.assertAlmostEqual(58, df3.shape[0], delta=7)

        # number of features should not change
        self.assertEqual(65, df1.shape[1])
        self.assertEqual(65, df2.shape[1])
        self.assertEqual(65, df3.shape[1])
        # three csv file in output
        num_csv = len(glob.glob(str(Path(out_path).joinpath("*.csv"))))
        self.assertEqual(4, num_csv)

    @unittest.skipIf(platform.system().lower() == 'windows', "Plotting too memory extensive. Skipping test!")
    def test_run_stack_no_nuclei(self):
        in_path = str(self.get_test_image_folder("n"))
        param_file = str(self.get_test_parameter_file("parameters_no_nuclei.yml"))
        out_path = str(self.output_path.joinpath("run_stack_no_nuclei"))

        # build arguments
        sys.argv = [sys.argv[0]] + ['run-stack', param_file, in_path, out_path]

        # call
        startup()

        # assert
        self.check_doc_output(out_path)

        # load csv
        df1 = pd.read_csv(Path(out_path).joinpath("MAX_8h_flow_uslide_new_setup_2021_10_14__21_19_47.csv"))
        df2 = pd.read_csv(Path(out_path).joinpath("MAX_8h_flow_uslide_new_setup_2021_10_14__21_29_39.csv"))
        df3 = pd.read_csv(Path(out_path).joinpath("MAX_8h_flow_uslide_new_setup_2021_10_14__21_36_01.csv"))

        # assert row count
        self.assertAlmostEqual(33, df1.shape[0], delta=5)
        self.assertAlmostEqual(28, df2.shape[0], delta=4)
        self.assertAlmostEqual(29, df3.shape[0], delta=4)

        # number of features should not change
        self.assertEqual(47, df1.shape[1])
        self.assertEqual(47, df2.shape[1])
        self.assertEqual(47, df3.shape[1])

        # three csv file in output
        num_csv = len(glob.glob(str(Path(out_path).joinpath("*.csv"))))
        self.assertEqual(4, num_csv)

    @unittest.skipIf(platform.system().lower() == 'windows', "Plotting too memory extensive. Skipping test!")
    def test_run_key(self):
        in_path = str(self.get_test_image_folder("gn"))
        in_key = str(self.get_test_key_file())
        param_file = str(self.get_test_parameter_file("parameters_golgi_nuclei.yml"))
        out_path = str(self.output_path.joinpath("run_key"))

        # build arguments
        sys.argv = [sys.argv[0]] + ['run-key', param_file, in_path, in_key, out_path]

        # call
        startup()

        # assert
        self.check_doc_output(out_path)

        # load summary csv
        df_sum = pd.read_csv(Path(out_path).joinpath("summary_table.csv"))

        # processed three images
        self.assertEqual(df_sum.shape, (3, 4))

        # processed two conditions
        Path(out_path).joinpath("cond_1", "merged_table_cond_1.csv").exists()
        Path(out_path).joinpath("cond_2", "merged_table_cond_2.csv").exists()
