import time
import unittest
from pathlib import Path

import polarityjam.test.test_config as config
import polarityjam.test.test_feature_extraction as test_feature_extraction
import polarityjam.test.test_integration as test_integration

from polarityjam.polarityjam_logging import get_logger
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
    suite.addTests(loader.loadTestsFromModule(test_integration))

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
