import time
import unittest
from pathlib import Path

import polarityjam.test.test_config as config
from polarityjam.polarityjam_logging import get_logger
from polarityjam.test.integration import test_commandline, test_masks, test_properties
from polarityjam.test.unit import test_feature_extraction
from polarityjam.utils.io import create_path_recursively


def start_tests(target_folder=None):
    if target_folder:
        config._TARGET_DIR = Path(target_folder)
        create_path_recursively(target_folder)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    # unittests
    suite.addTests(loader.loadTestsFromModule(test_feature_extraction))

    # integration tests
    suite.addTests(loader.loadTestsFromModule(test_commandline))
    suite.addTests(loader.loadTestsFromModule(test_masks))
    suite.addTests(loader.loadTestsFromModule(test_properties))

    runner = unittest.TextTestRunner(verbosity=3)
    result = runner.run(suite)
    if result.wasSuccessful():
        time.sleep(5)
        get_logger().info("Success")
        exit(0)
    else:
        get_logger().warning("Failed")
        exit(1)


if __name__ == "__main__":
    start_tests()
