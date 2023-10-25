"""Module holding the deepcell segmentation solution."""
from album.runner.api import get_args, setup

# DO NOT IMPORT ANYTHING OTHER THAN FROM RUNNER API HERE


env_file = """name:  DeepCell
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.9
  - tifffile>=2023.7.18
  - cudatoolkit
  - cudnn
  - pip
  - pip:
    - deepcell==0.12.7
"""


def setup_album():
    """Initialize the album api."""
    from album.api import Album

    album_api = Album.Builder().build()
    album_api.load_or_create_collection()

    return album_api


def run():
    """Python code that is executed in the solution environment."""
    # parse arguments
    args = get_args()

    # imports
    import argparse
    from pathlib import Path

    import numpy as np
    import tifffile
    from deepcell.applications import Mesmer
    from skimage.io import imread

    def segmentation(
        args: argparse.Namespace, image_nuclear: str, image_membrane: str
    ) -> str:
        """Load the color channels, prepares them for the network and runs the segmentation.

        Args:
            args: Arguments provided by the user.
            image_nuclear: Name of the image containing the nuclei.
            image_membrane: Name of the image containing the membrane.

        Returns:
            Name of the processed image.
        """
        image_nuc = imread(Path(args.folder_path) / image_nuclear)
        image_mem = imread(Path(args.folder_path) / image_membrane)
        if len(image_nuc.shape) > 2:
            image_nuc = image_nuc[:, :, args.channel_nuclear]
        if len(image_mem.shape) > 2:
            image_mem = image_mem[:, :, args.channel_membrane]
        # if its only one channel (H and W), add a channel dimension
        if len(image_nuc.shape) == 2:
            image_nuc = np.expand_dims(image_nuc, axis=-1)
        if len(image_mem.shape) == 2:
            image_mem = np.expand_dims(image_mem, axis=-1)

        # combine and expand to [1, H, W, C]
        image_prepped = np.concatenate([image_nuc, image_mem], axis=-1)
        image_prepped = np.expand_dims(image_prepped, 0)

        # model loading
        # see https://github.com/vanvalenlab/deepcell-tf/blob/master/deepcell/applications/mesmer.py
        app = Mesmer()
        print(
            "Image resolution the network was trained on:",
            app.model_mpp,
            "microns per pixel",
        )

        # prediction
        segmentation_predictions = app.predict(
            image_prepped,
            image_mpp=args.image_mpp,
            compartment=args.segmentation_mode,
            batch_size=1,
        )
        segmentation_predictions = np.squeeze(segmentation_predictions)
        # convert from 64 to 16 bit
        segmentation_predictions = segmentation_predictions.astype(np.uint16)

        # save the segmentation predictions
        image_name = Path(image_nuclear).stem
        output_name = Path(args.output_path) / (image_name + "_segmentation")
        Path(args.output_path).mkdir(parents=True, exist_ok=True)
        if args.save_mask:
            tifffile.imwrite(output_name.with_suffix(".tiff"), segmentation_predictions)
        if args.save_npy:
            np.save(str(output_name) + ".npy", segmentation_predictions)
        return str(output_name)

    # test if folder_path contains tif files
    folder_path = Path(args.folder_path)
    if folder_path.is_dir():
        # get all files in folder
        files = folder_path.glob("*.tif*")
        if not files:
            raise ValueError("No tif files found in folder.")
    else:
        raise ValueError("The provided folder path is not a folder.")

    output_name = None
    # run on one sample or whole folder
    if args.image_nuclear and args.image_membrane:
        output_name = segmentation(args, args.image_nuclear, args.image_membrane)
    else:
        for file in files:
            output_name = segmentation(args, file, file)

    if not output_name:
        raise ValueError(
            "Nothing done! No input provided (%s, %s)."
            % (args.image_nuclear, args.image_membrane)
        )

    print("Segmentation saved to:", str(Path(output_name).resolve()))


setup(
    group="polarityjam",
    name="DeepCell-predict",
    version="0.1.0",
    title="DeepCell Mesmer Segmentation",
    description="A solution to create segmentations with the previously trained deepCell Mesmer model",
    solution_creators=["Maximilian Otto", "Jan Philipp Albrecht"],
    tags=["segmentation", "machine_learning", "images", "deepcell", "mesmer", "2D"],
    license="Modified Apache v2",
    documentation=["https://deepcell.readthedocs.io/en/master/"],
    covers=[],
    album_api_version="0.5.5",
    args=[
        {
            "name": "image_nuclear",
            "type": "string",
            "required": False,
            "description": "Name of image containing the nuclei.",
            "default": "",
        },
        {
            "name": "channel_nuclear",
            "type": "integer",
            "required": True,
            "description": "Channel index of the nuclear image, starting with 0.",
            "default": 0,
        },
        {
            "name": "image_membrane",
            "type": "string",
            "required": False,
            "description": "Name of the image containing the membrane."
            " This can be the same file as the nuclear image,"
            " but would require to set `channel_membrane` to a different value.",
            "default": "",
        },
        {
            "name": "channel_membrane",
            "type": "integer",
            "required": True,
            "description": "Channel index of the membrane image, starting with 0.",
            "default": 0,
        },
        {
            "name": "image_mpp",
            "type": "float",
            "required": False,
            "description": "Resolution of the image in `Microns per pixel`."
            " If not provided, the resolution of the model will be used.",
            "default": 0.5,
        },
        {
            "name": "segmentation_mode",
            "type": "string",
            "required": False,
            "description": "Segmentation mode can be either `whole-cell` or `nuclear`.",
            "default": "whole-cell",
        },
        {
            "name": "output_path",
            "type": "string",
            "required": False,
            "description": "Full path to the output folder in which the result gets stored.",
            "default": "./output",
        },
        {
            "name": "save_mask",
            "type": "boolean",
            "required": False,
            "description": "Set this to `True` to save the segmentation mask as a tif file.",
            "default": True,
        },
        {
            "name": "save_npy",
            "type": "boolean",
            "required": False,
            "description": "Set this to `True` to save the segmentation mask as a numpy file.",
            "default": False,
        },
        {
            "name": "folder_path",
            "type": "string",
            "required": True,
            "description": "Full path to the folder containing the images."
            " If provided without the names of the images, the solution will run on all images in "
            "the folder while assuming each image contains nuclei and membrane"
            " information in the provided color channels.",
            "default": "./",
        },
    ],
    run=run,
    dependencies={"environment_file": env_file},
)


class DeepCellSegmenter:
    """Polyrityjam Segmenter class for DeepCell Mesmer."""

    def __init__(self, params):
        """Initialize the segmenter with the given parameters."""
        self.params = params
        self.img_path = None
        self.mpp = 1
        self.tmp_dir = None

    def segment(
        self,
        img,
        path=None,
        mode=None,
    ):
        """Segment the given image."""
        import os
        import tempfile

        import numpy as np
        from tifffile import tifffile

        from polarityjam.controller.segmenter import SegmentationMode
        from polarityjam.polarityjam_logging import get_logger

        def _install():
            # path to this file
            path = os.path.abspath(__file__)
            album.install(path)

        if path is not None:
            get_logger().warning(
                "You configured a path for loading an existing segmentation. "
                "This segmentation algorithm does not support loading segmentations from disk!"
            )

        if mode is None:
            mode = SegmentationMode.CELL

        if isinstance(mode, str):
            try:
                mode = SegmentationMode(mode)
            except ValueError:
                raise ValueError(
                    'Mode must be either "nucleus", "organelle", "cell" or "junction".'
                )

        if mode == SegmentationMode.JUNCTION:
            raise ValueError("This segmentation algorithm does not support this mode!")
        elif mode == SegmentationMode.ORGANELLE:
            get_logger().info(
                "This model is probably not trained for organelles segmentation. Please handle results with care."
            )
        elif mode == SegmentationMode.CELL:
            get_logger().info("Start segmentation procedure for cells...")
            self.params.segmentation_mode = "whole-cell"
        elif mode == SegmentationMode.NUCLEUS:
            get_logger().info("Start segmentation procedure for nuclei...")
            self.params.segmentation_mode = "nuclear"
        else:
            raise ValueError(
                'Mode must be either "nucleus", "organelle", "cell" or "junction".'
            )

        solution_id = "polarityjam:DeepCell-predict:0.1.0"
        album = setup_album()

        try:
            r = album.is_installed(solution_id)
            if not r:
                _install()
        except LookupError:
            _install()

        # save img to temporary folder
        self.tmp_dir = tempfile.TemporaryDirectory(dir=tempfile.gettempdir())
        self.img_path = os.path.join(self.tmp_dir.name, "segmentation.tif")
        tifffile.imwrite(self.img_path, img)

        # build arguments
        argv = [
            os.path.dirname(os.path.realpath(__file__)),
            "--channel_nuclear=%s" % 1,
            "--channel_membrane=%s" % 0,
            "--save_npy=%s" % "True",
            "--segmentation_mode=%s" % self.params.segmentation_mode,
            "--save_mask=%s" % self.params.save_mask,
            "--image_mpp=%s" % self.mpp,
            "--output_path=%s" % self.tmp_dir.name,
            "--folder_path=%s" % self.tmp_dir.name,
        ]

        # call solution
        album.run(solution_to_resolve=solution_id, argv=argv)

        # load segmentation
        seg = np.load(os.path.join(self.tmp_dir.name, "segmentation_segmentation.npy"))

        return seg

    def prepare(self, img, input_parameter):
        """Prepare the image for segmentation."""
        import numpy as np

        from polarityjam.model.parameter import ImageParameter

        # store parameters
        self.mpp = 1 / input_parameter.pixel_to_micron_ratio

        if (
            input_parameter.channel_nucleus == -1
            or input_parameter.channel_nucleus is None
        ):
            raise ValueError("Segmentation without nucleus channel is not supported!")

        if (
            input_parameter.channel_junction == -1
            or input_parameter.channel_junction is None
        ):
            raise ValueError("Segmentation without nucleus channel is not supported!")

        img_s = img
        if img.shape[0] < img.shape[-1]:
            img_s = np.einsum("kij->ijk", img)  # channel last

        img_s = np.dstack(
            (
                img_s[:, :, input_parameter.channel_junction],
                img_s[:, :, input_parameter.channel_nucleus],
                np.zeros((img_s.shape[0], img_s.shape[1])),
            )
        )

        # build prepared image parameters
        params_prep_img = ImageParameter()
        params_prep_img.reset()
        params_prep_img.channel_junction = 0
        params_prep_img.channel_nucleus = 1
        params_prep_img.pixel_to_micron_ratio = input_parameter.pixel_to_micron_ratio

        return img_s, params_prep_img

    def __del__(self):
        """Clean up the temporary directory."""
        if self.tmp_dir is not None:
            self.tmp_dir.cleanup()
