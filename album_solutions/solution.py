"""polarityjam album solution."""
from album.runner.api import get_args, get_package_path, setup

env_file = """name: Polarityjam
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.8.13
  - git=2.34
  - pip
  - pip:
     - numpy==1.21.5
     - cellpose==1.0.1
     - pandas==1.4.1
     - scipy==1.8.0
     - scikit-image==0.19.2
     - matplotlib==3.5.1
     - PyYAML==6.0
     - cmocean==2.0
     - pyunpack==0.2.2
     - networkx==2.6.3
     - GitPython==3.1
"""


def install():
    """Install the package."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    from git import Repo

    print("Installing polarityjam and dependencies")
    polarityjam_repo = Path(get_package_path()).joinpath("polarityjam")
    os.mkdir(polarityjam_repo)
    # todo: point to commit hash
    Repo.clone_from("https://github.com/polarityjam/polarityjam.git", polarityjam_repo)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-e", Path(polarityjam_repo)]
    )


def run():
    """Run the package."""
    import os

    run_string = str(get_args().run_command)
    parameter_string = str(get_args().parameter_file)
    input_string = str(get_args().input_file)
    output_string = str(get_args().output_path)
    filename_prefix_string = str(get_args().filename_prefix)
    key_string = str(get_args().key_csv)

    polarityjam_string = "polarityjam"
    polarityjam_call = "polarityjam_test"

    # Determine the mode to run polarityjam
    if run_string != "test":
        # Normal run
        if run_string == "run":
            # run with prefix
            if filename_prefix_string != "None":
                polarityjam_call = "{pjam} {run} {param} {input} {output} --filename_prefix {prefix}".format(
                    pjam=polarityjam_string,
                    run=run_string,
                    param=parameter_string,
                    input=input_string,
                    output=output_string,
                    prefix=filename_prefix_string,
                )

            # run without prefix
            else:
                polarityjam_call = "{pjam} {run} {param} {input} {output}".format(
                    pjam=polarityjam_string,
                    run=run_string,
                    param=parameter_string,
                    input=input_string,
                    output=output_string,
                )

        # run with keys csv
        if run_string == "run-key":
            if key_string != "None":
                polarityjam_call = "{pjam} {run} {param} {input} {key} {output}".format(
                    pjam=polarityjam_string,
                    run=run_string,
                    param=parameter_string,
                    input=input_string,
                    key=key_string,
                    output=output_string,
                )
            else:
                print(
                    "You tried to run polarityjam in the key mode without providing a key.csv file!"
                )

        # Run on multiple files
        if run_string == "run-stack":
            polarityjam_call = "{pjam} {run} {param} {input} {output}".format(
                pjam=polarityjam_string,
                run=run_string,
                param=parameter_string,
                input=input_string,
                output=output_string,
            )

    # start polarityjam
    os.system(polarityjam_call)


def prepare_test():
    """Prepare the test."""
    return {}


def test():
    """Run the test."""
    import os

    os.system("polarityjam_test")


setup(
    group="de.mdc-berlin",
    name="polarityjam",
    version="0.1.0",
    title="A Solution to run the polarityjam feature extraction pipeline",
    description="A Solution to run the polarityjam Feature Extraction Pipeline.",
    solution_creators=["Lucas Rieckert", "Jan Philipp Albrecht"],
    tags=["polarityjam", "test"],
    license="UNLICENSE",
    documentation=["doc.md"],
    covers=[{"description": "Polarityjam cover image", "source": "cover.png"}],
    album_api_version="0.5.1",
    args=[
        {
            "name": "run_command",
            "type": "string",
            "default": "run",
            "description": """How do you want to run polarityjam? run for single tiff file,
            run-stack for a directory containing multiple tiff files, run-key for a csv
            file containing a list of  directorys containing tiff files or test for the test-suit""",
            "required": True,
        },
        {
            "name": "input_file",
            "type": "file",
            "description": "Path to the input tiff file (or directory of tiff files for run-stack).",
            "required": True,
        },
        {
            "name": "parameter_file",
            "type": "file",
            "description": "Path to the parameter file.",
            "required": True,
        },
        {
            "name": "output_path",
            "type": "string",
            "description": "Path to the directory where the output will get saved.",
            "required": True,
        },
        {
            "name": "filename_prefix",
            "type": "string",
            "description": "Optional file prefix to rename the input files.",
        },
        {
            "name": "key_csv",
            "type": "file",
            "description": "CSV file containing the keys for the input.",
        },
    ],
    install=install,
    run=run,
    pre_test=prepare_test,
    test=test,
    dependencies={"environment_file": env_file},
)
