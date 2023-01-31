from pathlib import Path

import networkx as nx
import numpy as np
import yaml

from polarityjam.model.moran import run_morans
from polarityjam.test.test_common import TestCommon
from polarityjam.utils.io import read_parameters, read_image


class TestFunctions(TestCommon):
    """Small tests to ensure basic IO functionality of the tool."""

    def setUp(self) -> None:
        super().setUp()

    def test_read_parameters(self):
        # prepare
        local_parameter_file = {
            "channel_junction": 0,
            "channel_nucleus": 1,
            "channel_organelle": 2
        }
        local_parameter_file_path = Path(self.tmp_dir).joinpath("local_parameter_file.yml")

        with open(local_parameter_file_path, "w+") as file:
            file.write(yaml.dump(local_parameter_file, Dumper=yaml.Dumper))

        self.parameters["channel_junction"] = 0
        self.parameters["channel_nucleus"] = 1
        self.parameters["channel_organelle"] = 2

        # call
        r = read_parameters(local_parameter_file_path)

        # assert
        self.assertDictEqual(self.parameters, r)

    def test_read_img(self):
        # prepare
        test_image_path = self.get_test_image_path("060721_EGM2_18dyn_01.tif")

        # call
        r = read_image(test_image_path)

        # assert
        self.assertIsNotNone(r)

        # assert correct type
        self.assertEqual(type(np.array(0)), type(r))

    def test_morans_i_anti_correlation(self):
        # prepare anti-correlation sample
        graph = nx.grid_2d_graph(10, 10)

        for i in range(0, 10):
            if (i % 2) == 0:
                for j in range(0, 10):
                    if (j % 2) == 0:
                        nx.set_node_attributes(graph, {(i, j): 0}, name="morani")
                    else:
                        nx.set_node_attributes(graph, {(i, j): 1}, name="morani")
            else:
                for j in range(0, 10):
                    if (j % 2) == 0:
                        nx.set_node_attributes(graph, {(i, j): 1}, name="morani")

                    else:
                        nx.set_node_attributes(graph, {(i, j): 0}, name="morani")

        # run morans i
        mr_i = run_morans(graph, "morani")

        # check the output
        self.assertEqual(-1, mr_i.I)

    def test_morans_i_random(self):
        # prepare random sample
        graph = nx.grid_2d_graph(10, 10)

        empty_int = np.zeros((10, 10))
        for i in range(0, 10):
            for j in range(0, 10):
                rand_int = np.random.normal(0, 1)
                if rand_int >= .5:
                    empty_int[i, j] = 1
                else:
                    empty_int[i, j] = 0
                nx.set_node_attributes(graph, {(i, j): rand_int}, name="morani")
        # run
        mr_i = run_morans(graph, "morani")

        # assert
        self.assertAlmostEqual(0, mr_i.I, delta=0.3)

    def test_morans_i_correlated(self):
        # prepare correlated sample
        graph = nx.grid_2d_graph(10, 10)

        empty_int = np.zeros((10, 10))
        for i in range(0, 10):
            for j in range(0, 10):
                rand_int = 1.1
                empty_int[i, j] = rand_int
                nx.set_node_attributes(graph, {(i, j): rand_int}, name="morani")

        # run
        mr_i = run_morans(graph, "morani")

        # assert
        self.assertEqual(1, mr_i.I)


class EmptyTestClass:
    pass
