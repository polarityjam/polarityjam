import fileinput
import importlib.util
import os
import sys
from pathlib import Path

from nbconvert import PythonExporter

from polarityjam.test.test_common import TestCommon


class TestIntegration(TestCommon):
    def nb_to_py(self, path_nb, filename, outpath_py, prefix_str="tmp_"):
        outpath_py = Path(outpath_py)

        # export
        exp = PythonExporter()
        nb_python_mem, _ = exp.from_filename(str(path_nb))

        p = outpath_py.joinpath(prefix_str + filename + ".py")
        with open(p, "w+") as f:
            f.write(nb_python_mem)

        return p

    def test_nb(self):
        # copy to notebook tmp dir
        path_nb = Path(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent.joinpath(
            "notebooks", "polarityjam-notebook.ipynb"
        )
        p = self.nb_to_py(path_nb, path_nb.stem, self.tmp_dir)

        # remove ipython formatting and set data path
        replace_cell_pattern = "### DELETE ME ###"
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
                        while not next(myiter, "None").startswith(replace_cell_pattern):
                            print("")
                    else:
                        print(line, end='')
                else:
                    print(line, end='')

        root = Path(p).parent
        sys.path.insert(0, str(root))
        os.chdir(root)
        spec = importlib.util.spec_from_file_location("nb_test", p)
        nb_test = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(nb_test)

    def setUp(self) -> None:
        super().setUp()
        if not self.data_path.exists():
            self.extract_test_data()