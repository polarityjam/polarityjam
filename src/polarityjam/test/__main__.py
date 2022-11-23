import argparse

from polarityjam.test.test_all import start_tests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--target-folder',
        required=False,
        help='Specify a folder where the test output is written to. If none is given, temporary folder is taken.',
        default=None
    )
    args = parser.parse_args()

    start_tests(args.target_folder)


if __name__ == "__main__":
    main()
