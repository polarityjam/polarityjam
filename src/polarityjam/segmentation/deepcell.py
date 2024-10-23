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
    import os
    from pathlib import Path

    from album.api import Album

    # create a collection inside the solution calling "album in album"
    album_base_path = Path(os.path.abspath(__file__)).parent
    album_api = Album.Builder().base_cache_path(album_base_path).build()
    album_api.load_or_create_collection()

    return album_api


def run():
    """Run the deepcell solution."""
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
        args: argparse.Namespace, image_nuclear, image_membrane=None
    ) -> str:
        """Load the color channels, prepares them for the network, and runs the segmentation.

        Args:
            args: Arguments provided by the user.
            image_nuclear: Name of the image containing the nuclei.
            image_membrane: Name of the image containing the membrane (optional).

        Returns:
            Name of the processed image.
        """
        image_nuc = imread(Path(args.input_path) / image_nuclear)

        if len(image_nuc.shape) > 2:
            image_nuc = image_nuc[:, :, args.channel_nuclear]
        if len(image_nuc.shape) == 2:
            image_nuc = np.expand_dims(image_nuc, axis=-1)

        if image_membrane:
            image_mem = imread(Path(args.input_path) / image_membrane)
            if len(image_mem.shape) > 2:
                image_mem = image_mem[:, :, args.channel_membrane]
            if len(image_mem.shape) == 2:
                # Expand from [H, W] to [H, W, C]
                image_mem = np.expand_dims(image_mem, axis=-1)
        else:
            # Create an empty channel for membrane if it's not provided
            image_mem = np.zeros_like(image_nuc)
            args.segmentation_mode = "nuclear"

        # Combine and expand to [1, H, W, C]
        image_prepped = np.concatenate([image_nuc, image_mem], axis=-1)
        image_prepped = np.expand_dims(image_prepped, 0)

        # Model loading
        # see https://github.com/vanvalenlab/deepcell-tf/blob/master/deepcell/applications/mesmer.py
        app = Mesmer()
        print(
            "Image resolution the network was trained on:",
            app.model_mpp,
            "microns per pixel",
        )

        # Postprocessing
        if (args.maxima_threshold is None) or (args.maxima_threshold <= 0):
            if args.segmentation_mode == "whole-cell":
                args.maxima_threshold = 0.1
            elif args.segmentation_mode == "nuclear":
                args.maxima_threshold = 0.075

        postprocess = {
            "maxima_threshold": args.maxima_threshold,
            "maxima_smooth": args.maxima_smooth,
            "interior_threshold": args.interior_threshold,
            "interior_smooth": args.interior_smooth,
            "small_objects_threshold": args.small_objects_threshold,
            "fill_holes_threshold": args.fill_holes_threshold,
            "radius": args.radius,
            "pixel_expansion": args.pixel_expansion,
        }

        # Prediction
        segmentation_predictions = app.predict(
            image_prepped,
            image_mpp=args.image_mpp,
            postprocess_kwargs_whole_cell=postprocess,
            postprocess_kwargs_nuclear=postprocess,
            compartment=args.segmentation_mode,
            batch_size=1,
        )
        segmentation_predictions = np.squeeze(segmentation_predictions)
        # Convert from 64 to 16 bit
        segmentation_predictions = segmentation_predictions.astype(np.uint16)

        # Save the segmentation predictions
        image_name = Path(image_nuclear).stem
        output_name = Path(args.output_path) / (image_name + "_segmentation")
        Path(args.output_path).mkdir(parents=True, exist_ok=True)
        if args.save_mask:
            tifffile.imwrite(output_name.with_suffix(".tiff"), segmentation_predictions)
        if args.save_npy:
            np.save(output_name, segmentation_predictions)
        return str(output_name)

    # Test if input_path contains tif files
    input_path = Path(args.input_path)
    if not input_path.is_dir():
        raise ValueError("The provided folder path is not a folder.")

    # Run on one sample or whole folder
    if args.img_name_nuclear:
        output_name = segmentation(args, args.img_name_nuclear, args.img_name_membrane)
    else:
        files = list(input_path.glob("*.tif*"))
        if not files:
            raise ValueError("No tif files found in folder.")
        for file in files:
            output_name = segmentation(args, file, file)

    print("Recent segmentation saved to:", Path(output_name).resolve())
    return


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
    cite=[
        {
            "text": "Noah F. Greenwald and Geneva Miller and Erick Moen and Alex Kong and Adam Kagel and Thomas Dougherty and Christine Camacho Fullaway and Brianna J. McIntosh and Ke Xuan Leow and Morgan Sarah Schwartz and Cole Pavelchek and Sunny Cui and Isabella Camplisson and Omer Bar-Tal and Jaiveer Singh and Mara Fong and Gautam Chaudhry and Zion Abraham and Jackson Moseley and Shiri Warshawsky and Erin Soon and Shirley Greenbaum and Tyler Risom and Travis Hollmann and Sean C. Bendall and Leeat Keren and William Graf and Michael Angelo and David Van Valen; Whole-cell segmentation of tissue images with human-level performance using large-scale data annotation and deep learning",  # noqa: E501
            "doi": "https://doi.org/10.1038/s41587-021-01094-0",
        }
    ],
    covers=[],
    album_api_version="0.5.5",
    args=[
        {
            "name": "input_path",
            "type": "string",
            "required": True,
            "description": "Full path to the folder containing the images.",
            "default": "./",
        },
        {
            "name": "img_name_nuclear",
            "type": "string",
            "required": False,
            "description": "Name of image containing the nuclei. (If not set or empty, the solution will run on all images in the folder while assuming each image contains nuclei and membrane information in the provided color channels.)",  # noqa: E501
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
            "name": "img_name_membrane",
            "type": "string",
            "required": False,
            "description": "Name of the image containing the membrane information. This can be in the same file as the nuclear image, but would require to set `channel_membrane` to a different value. If not set, an empty channel will be used for the membrane.",  # noqa: E501
            "default": "",
        },
        {
            "name": "channel_membrane",
            "type": "integer",
            "required": False,
            "description": "Channel index of the membrane image, starting with 0. If `img_name_membrane` is not set, this value will be ignored.",  # noqa: E501
            "default": 1,
        },
        {
            "name": "image_mpp",
            "type": "float",
            "required": False,
            "description": "Resolution of the images in `Microns per pixel`. If not set, it will default to 0.5.",
            "default": 0.5,
        },
        {
            "name": "segmentation_mode",
            "type": "string",
            "required": False,
            "description": "Segmentation mode can be either `whole-cell` or `nuclear`. If `img_name_membrane` is not set, this will default to `nuclear`.",  # noqa: E501
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
            "name": "maxima_threshold",
            "type": "float",
            "required": False,
            "description": "To finetune specific and consistent errors in your data, the following arguments can be used during postprocessing. Lower values will result in more cells being detected. Higher values will result in fewer cells being detected. When set to -1, the default for mode `whole_cell` (0.1) and for `nuclear` (0.075) will be applied",  # noqa: E501
        },
        {
            "name": "maxima_smooth",
            "type": "float",
            "required": False,
            "description": "Default: 0",
            "default": 0.0,
        },
        {
            "name": "interior_threshold",
            "type": "float",
            "required": False,
            "description": "Comparison threshold for cell vs. background. Lower values tend to result in larger cells. Default: 0.2",  # noqa: E501
            "default": 0.2,
        },
        {
            "name": "interior_smooth",
            "type": "float",
            "required": False,
            "description": "Default: 0",
            "default": 0.0,
        },
        {
            "name": "small_objects_threshold",
            "type": "float",
            "required": False,
            "description": "Default: 15",
            "default": 15.0,
        },
        {
            "name": "fill_holes_threshold",
            "type": "float",
            "required": False,
            "description": "Default: 15",
            "default": 15.0,
        },
        {
            "name": "radius",
            "type": "float",
            "required": False,
            "description": ". Default: 2",
            "default": 2.0,
        },
        {
            "name": "pixel_expansion",
            "type": "integer",
            "required": False,
            "description": "Add a manual pixel expansion after segmentation to each cell. Default: 0",
            "default": 0,
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
            "--input_path=%s" % self.tmp_dir.name,
            "--maxima_threshold=%s" % self.params.maxima_threshold,
            "--maxima_smooth=%s" % self.params.maxima_smooth,
            "--interior_threshold=%s" % self.params.interior_threshold,
            "--interior_smooth=%s" % self.params.interior_smooth,
            "--small_objects_threshold=%s" % self.params.small_objects_threshold,
            "--fill_holes_threshold=%s" % self.params.fill_holes_threshold,
            "--radius=%s" % self.params.radius,
            "--pixel_expansion=%s" % self.params.pixel_expansion,
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

        # check which channel is configured to use for cell segmentation:
        channel_cell_segmentation = input_parameter.channel_junction
        if self.params.channel_cell_segmentation != "":
            try:
                channel_cell_segmentation = input_parameter.__getattribute__(
                    self.params.channel_cell_segmentation
                )
            except AttributeError as e:
                raise AttributeError(
                    "Channel %s does not exist! Wrong segmentation configuration!"
                    % self.params.channel_cell_segmentation
                ) from e

        # check which channel is configured to use for nuclei segmentation:
        channel_nuclei_segmentation = input_parameter.channel_nucleus
        if self.params.channel_nuclei_segmentation != "":
            try:
                channel_nuclei_segmentation = input_parameter.__getattribute__(
                    self.params.channel_nuclei_segmentation
                )
            except AttributeError as e:
                raise AttributeError(
                    "Channel %s does not exist! Wrong segmentation configuration!"
                    % self.params.channel_nuclei_segmentation
                ) from e

        if channel_nuclei_segmentation == -1 or channel_nuclei_segmentation is None:
            raise ValueError("Segmentation without nucleus channel is not supported!")

        if channel_cell_segmentation == -1 or channel_cell_segmentation is None:
            raise ValueError("Segmentation without junction channel is not supported!")

        img_s = img
        if img.shape[0] < img.shape[-1]:
            img_s = np.einsum("kij->ijk", img)  # channel last

        img_s = np.dstack(
            (
                img_s[:, :, channel_cell_segmentation],
                img_s[:, :, channel_nuclei_segmentation],
                np.zeros((img_s.shape[0], img_s.shape[1])),
            )
        )

        # build prepared image parameters
        params_prep_img = ImageParameter()
        params_prep_img.reset()
        params_prep_img.channel_junction = (
            0  # might be called junction channel, but content depends on config
        )
        params_prep_img.channel_nucleus = (
            1  # might be called nuclei channel, but content depends on config
        )
        params_prep_img.pixel_to_micron_ratio = input_parameter.pixel_to_micron_ratio

        return img_s, params_prep_img

    def __del__(self):
        """Clean up the temporary directory."""
        if self.tmp_dir is not None:
            self.tmp_dir.cleanup()
