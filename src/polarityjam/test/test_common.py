import os
import tempfile
import unittest
from pathlib import Path

import yaml
from pyunpack import Archive

import polarityjam.test.test_config as config
from polarityjam.polarityjam_logging import close_logger
from polarityjam.utils.io import list_files_recursively


class TestCommon(unittest.TestCase):
    """Base class all tests inherit from. Responsible for temporary directory, cleanup, parameters."""

    def setUp(self) -> None:
        if config._TARGET_DIR:
            self.tmp_dir_ = Path(config._TARGET_DIR)
            self.tmp_dir = self.tmp_dir_
            self.output_path = self.tmp_dir.joinpath("output")
            self.data_path = self.tmp_dir.joinpath("data")
        else:
            self.tmp_dir_ = tempfile.TemporaryDirectory()
            self.tmp_dir = self.tmp_dir_.name
            self.output_path = Path(self.tmp_dir).joinpath("output")
            self.data_path = Path(self.tmp_dir).joinpath("data")

        self.current_path = Path(os.path.dirname(os.path.realpath(__file__)))
        self.parameters, self.param_base_file = self.load_parameters()

    def tearDown(self) -> None:
        close_logger()
        if isinstance(self.tmp_dir_, tempfile.TemporaryDirectory):
            self.tmp_dir_.cleanup()

    def extract_test_data(self):
        target = str(self.tmp_dir)
        Archive(str(self.current_path.joinpath("resources", "data.zip"))).extractall(str(target))

    def get_test_image_path(self, image_name):
        if not self.data_path.exists():
            self.extract_test_data()

        files_list = list_files_recursively(self.data_path)
        test_image_path = files_list[[i.stem + i.suffix for i in files_list].index(image_name)]

        return test_image_path

    def get_test_image_folder(self, mode):
        # "gn" = golgi_nuclei, "g" = "no golgi", "n" = "no nuclei"

        if not self.data_path.exists():
            self.extract_test_data()

        if mode == "gn":
            return self.data_path.joinpath("golgi_nuclei")
        elif mode == "n":
            return self.data_path.joinpath("no_nuclei")
        elif mode == "g":
            return self.data_path.joinpath("no_golgi")
        else:
            return None

    def get_test_parameter_file(self, name):
        return self.current_path.joinpath("resources", name)

    def get_test_key_file(self):
        return self.current_path.joinpath("resources", "test_key_file.csv")

    def load_parameters(self):
        param_base_file = Path(self.current_path).joinpath("..", "utils", "resources", "parameters.yml")

        with open(param_base_file, 'r') as yml_f:
            parameters = yaml.safe_load(yml_f)

        parameters["channel_junction"] = 3
        parameters["channel_nucleus"] = 2
        parameters["channel_organelle"] = 0

        return parameters, param_base_file
