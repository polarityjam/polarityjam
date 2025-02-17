"""Argument parsing for the Polarityjam framework."""
import argparse
import os
import sys
from importlib.metadata import entry_points
from typing import Callable

import polarityjam
from polarityjam.commandline import run, run_key, run_stack
from polarityjam.polarityjam_logging import (
    close_logger,
    configure_logger,
    get_log_file,
    get_logger,
)
from polarityjam.utils.io import create_path_recursively


def startup():
    """Entry points of `polarityjam`."""
    parser = _create_parser()
    args, _ = parser.parse_known_args()

    __run_subcommand(args, parser)


def __run_subcommand(args, parser):
    """Call a specific subcommand."""
    command = ""
    try:
        command = sys.argv[1]  # command always expected at second position
    except IndexError:
        parser.error("Please provide a valid action!")

    if "out_path" not in args:
        args.out_path = os.getcwd()

    create_path_recursively(args.out_path)

    log_file = get_log_file(args.out_path)
    configure_logger("INFO", logfile_name=log_file)

    get_logger().debug("Running %s subcommand..." % command)

    get_logger().info("Polarityjam Version %s" % polarityjam.__version__)

    args.func(args)  # execute entry point function

    close_logger()


def _create_parser():
    """Create a parser for all known album arguments."""
    parser = PolarityjamParser()
    parser_creators = []

    for entry_point in entry_points(group="console_parsers_polarityjam"):
        try:
            parser_creators.append(entry_point.load())
        except Exception as e:
            get_logger().error("Cannot load console parser %s" % entry_point.name)
            get_logger().debug(str(e))
    for parse_creator in parser_creators:
        parse_creator(parser)
    return parser.parser


def create_run_parser(parser):
    """Create a parser for the `run` subcommand."""
    p = parser.create_file_command_parser(
        "run", run, "Feature extraction from a single tiff image."
    )
    p.add_argument("in_file", type=str, help="Path to the input tif file.")
    p.add_argument("out_path", type=str, help="Path to the output folder.")
    p.add_argument(
        "--filename_prefix",
        type=str,
        help="prefix for the output file.",
        required=False,
        default=None,
    )


def create_run_stack_parser(parser):
    """Create a parser for the `run-stack` subcommand."""
    p = parser.create_file_command_parser(
        "run-stack",
        run_stack,
        "Feature extraction from an input folder containing several tiff files.",
    )
    p.add_argument(
        "in_path", type=str, help="Name for the input folder containing tif images."
    )
    p.add_argument("out_path", type=str, help="Path to the output folder.")


def create_run_key_parser(parser):
    """Create a parser for the `run-key` subcommand."""
    p = parser.create_file_command_parser(
        "run-key",
        run_key,
        "Feature extraction from a given list of folders provided within a csv file.",
    )
    p.add_argument(
        "in_path",
        type=str,
        help="Path prefix. Note: Base path for all folders listed in the key file.",
    )
    p.add_argument(
        "in_key",
        type=str,
        help="Path to the input key file. File must contain column "
        '"folder_name" and "condition_name". Note: Folder name relative to the path prefix.',
    )
    p.add_argument("out_path", type=str, help="Path to the output folder.")


class ArgumentParser(argparse.ArgumentParser):
    """Override default error method of all parsers to show help of sub-command."""

    def error(self, message: str):
        """Print the error message and show the help of the sub-command."""
        self.print_help()
        self.exit(2, f"{self.prog}: error: {message}\n")


class PolarityjamParser(ArgumentParser):
    """Polarityjam argument parser."""

    def __init__(self):
        """Initialize the Polarityjam argument parser."""
        super().__init__()
        self.parent_parser = self.create_parent_parser()
        self.parser = self.create_parser()
        self.subparsers = self.parser.add_subparsers(
            title="actions", help="sub-command help"
        )

    @staticmethod
    def create_parent_parser() -> ArgumentParser:
        """Parent parser for all subparsers to have the same set of arguments."""
        parent_parser = ArgumentParser(add_help=False)
        parent_parser.add_argument(
            "--version", "-V", action="version", version="%s " % polarityjam.__version__
        )
        # parse logging
        parent_parser.add_argument(
            "--log",
            required=False,
            help="Logging level for your command. Choose between %s"
            % ", ".join(["INFO", "DEBUG"]),
            default="INFO",
        )
        return parent_parser

    def create_parser(self) -> ArgumentParser:
        """Create the main parser for the framework."""
        parser = ArgumentParser(
            add_help=True,
            description="Polarityjam - feature extraction pipeline for multichannel images.",
            parents=[self.parent_parser],
        )
        return parser

    def create_command_parser(
        self, command_name: str, command_function: Callable, command_help: str
    ) -> ArgumentParser:
        """Create a parser for a Polarityjam command, specified by a name, a function and a help description."""
        parser = self.subparsers.add_parser(
            command_name, help=command_help, parents=[self.parent_parser]
        )
        parser.set_defaults(func=command_function)
        return parser

    def create_file_command_parser(
        self, command_name: str, command_function: Callable, command_help: str
    ) -> ArgumentParser:
        """Create a parser for a Polarityjam command dealing with a file.

        Parser is specified by a name, a function and a help description.
        """
        parser = self.create_command_parser(
            command_name, command_function, command_help
        )
        parser.add_argument("param", type=str, help="Path to the parameter file.")
        return parser
