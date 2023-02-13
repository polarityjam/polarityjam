import fileinput
import importlib.util
import os
import platform
import sys
import unittest
from pathlib import Path

from nbconvert import PythonExporter

from polarityjam.test.test_common import TestCommon


class TestIntegration(TestCommon):
    def setUp(self) -> None:
        super().setUp()
        if not self.data_path.exists():
            self.extract_test_data()

    def tearDown(self) -> None:
        super().tearDown()

    def nb_to_py(self, path_nb, filename, outpath_py, prefix_str="tmp_"):
        outpath_py = Path(outpath_py)

        # export
        exp = PythonExporter()
        nb_python_mem, _ = exp.from_filename(str(path_nb))

        p = outpath_py.joinpath(prefix_str + filename + ".py")
        with open(p, "w+") as f:
            f.write(nb_python_mem)

        return p

    @unittest.skipIf(platform.system().lower() == 'windows', "Plotting too memory extensive. Skipping test!")
    def test_nb_2(self):
        # copy to notebook tmp dir
        path_nb = Path(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent.joinpath(
            "docs", "notebooks", "polarityjam-notebook_api.ipynb"
        )
        p = self.nb_to_py(path_nb, path_nb.stem, self.tmp_dir)

        # remove ipython formatting and set data path
        replace_cell_pattern = "### ADAPT ME ###"
        with fileinput.FileInput(p, inplace=True) as f:
            myiter = iter(f)
            for line in f:
                if line.startswith("get_ipython"):
                    print("")
                elif line.startswith("%"):
                    print("")
                elif line.startswith("!"):
                    print("")
                elif replace_cell_pattern:
                    if line.startswith(replace_cell_pattern):

                        print("input_file1 = Path(\"%s\")" % self.data_path.parent.joinpath(
                            "data", "golgi_nuclei", "set_2", "060721_EGM2_18dyn_02.tif"))
                        print("input_file2 = Path(\"%s\")" % self.data_path.parent.joinpath(
                            "data", "golgi_nuclei", "set_1", "060721_EGM2_18dyn_01.tif"))

                        print("output_path = Path(\"%s\")" % self.data_path.parent)
                        print("output_file_prefix1 = \"060721_EGM2_18dyn_02\"")
                        print("output_file_prefix2 = \"060721_EGM2_18dyn_01\"")

                        while not next(myiter, "None").startswith(replace_cell_pattern):
                            print("")
                    else:
                        print(line, end='')
                else:
                    print(line, end='')

        root = Path(p).parent
        sys.path.insert(0, str(root))
        spec = importlib.util.spec_from_file_location("nb_test2", p)
        nb_test2 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(nb_test2)

    @unittest.skipIf(platform.system().lower() == 'windows', "Plotting too memory extensive. Skipping test!")
    def test_nb_3(self):
        # copy to notebook tmp dir
        path_nb = Path(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent.joinpath(
            "docs", "notebooks", "polarityjam-notebook_vis.ipynb"
        )
        p = self.nb_to_py(path_nb, path_nb.stem, self.tmp_dir)

        # remove ipython formatting and set data path
        replace_cell_pattern = "### ADAPT ME ###"
        with fileinput.FileInput(p, inplace=True) as f:
            myiter = iter(f)
            for line in f:
                if line.startswith("get_ipython"):
                    print("")
                elif line.startswith("%"):
                    print("")
                elif line.startswith("!"):
                    print("")
                elif replace_cell_pattern:
                    if line.startswith(replace_cell_pattern):
                        print("output_path = Path(\"%s\")" % self.data_path.parent)
                        print("input_file = Path(\"%s\")" % self.data_path.parent.joinpath(
                            "data", "golgi_nuclei", "set_2", "060721_EGM2_18dyn_02.tif"))
                        print("output_file_prefix = \"060721_EGM2_18dyn_02\"")
                        while not next(myiter, "None").startswith(replace_cell_pattern):
                            print("")
                    else:
                        print(line, end='')
                else:
                    print(line, end='')

        root = Path(p).parent
        sys.path.insert(0, str(root))
        spec = importlib.util.spec_from_file_location("nb_test3", p)
        nb_test3 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(nb_test3)

    @unittest.skipIf(platform.system().lower() == 'windows', "Plotting too memory extensive. Skipping test!")
    def test_nb(self):
        # copy to notebook tmp dir
        path_nb = Path(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent.joinpath(
            "docs", "notebooks", "polarityjam-notebook.ipynb"
        )
        p = self.nb_to_py(path_nb, path_nb.stem, self.tmp_dir)

        # remove ipython formatting and set data path
        replace_cell_pattern = "### ADAPT ME ###"
        with fileinput.FileInput(p, inplace=True) as f:
            myiter = iter(f)
            for line in f:
                if line.startswith("get_ipython"):
                    print("")
                elif line.startswith("%"):
                    print("")
                elif line.startswith("!"):
                    print("")
                elif replace_cell_pattern:
                    if line.startswith(replace_cell_pattern):
                        print("output_path = Path(\"%s\")" % self.data_path.parent)
                        print("input_file = Path(\"%s\")" % self.data_path.parent.joinpath(
                            "data", "golgi_nuclei", "set_2", "060721_EGM2_18dyn_02.tif"))
                        print("output_file_prefix = \"060721_EGM2_18dyn_02\"")
                        while not next(myiter, "None").startswith(replace_cell_pattern):
                            print("")
                    else:
                        print(line, end='')
                else:
                    print(line, end='')

        root = Path(p).parent
        sys.path.insert(0, str(root))
        spec = importlib.util.spec_from_file_location("nb_test", p)
        nb_test = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(nb_test)
