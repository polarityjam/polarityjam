"""Module holding the SAM segmentation solution."""
from album.runner.api import get_active_logger, get_args, get_cache_path, setup

# DO NOT IMPORT ANYTHING OTHER THAN FROM RUNNER API HERE

env_file = """name:  microSAM
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.10
  - tifffile>=2023.7.18
  - pytorch>=2.0.0
  - torchvision>=0.15.0
  - pytorch-cuda=11.7
  - micro_sam=0.3.0post1
  - pip
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
    """Python code that is executed in the solution environment."""
    # parse arguments
    args = get_args()

    # cache path
    cache_path = get_cache_path()

    import os

    # inline backend from jupyter notebook does ot work
    _mpl_backend = os.environ.get("MPLBACKEND")
    if "backend_inline" in _mpl_backend:
        os.environ["MPLBACKEND"] = "qtagg"
        get_active_logger().info(
            f"Set MPLBACKEND from {_mpl_backend} to qtagg. Interactive needs to be disabled."
        )

    import numpy as np
    import tifffile
    from micro_sam import instance_segmentation, util
    from micro_sam.util import get_sam_model

    # open tiff
    img_channel = tifffile.TiffFile(args.input_path).asarray()

    embedding_path = args.embedding_path
    if (
        args.embedding_path is None
        or args.embedding_path == "None"
        or args.embedding_path == ""
    ):
        embedding_path = str(cache_path.joinpath("embeddings.zarr"))

    checkpoint_path = args.checkpoint_path
    if (
        args.checkpoint_path is None
        or args.checkpoint_path == "None"
        or args.checkpoint_path == ""
    ):
        checkpoint_path = None

    sam_pred = get_sam_model(
        model_type=args.model_name, checkpoint_path=checkpoint_path
    )
    amg = instance_segmentation.AutomaticMaskGenerator(sam_pred)
    embeddings = util.precompute_image_embeddings(
        sam_pred, img_channel, save_path=embedding_path
    )
    amg.initialize(img_channel, embeddings, verbose=True)
    instances_amg = amg.generate(pred_iou_thresh=args.pred_iou_thresh)
    instances_amg = instance_segmentation.mask_data_to_segmentation(
        instances_amg, shape=img_channel.shape, with_background=True
    )

    # save masks
    out_file = os.path.join(args.output_path, "mask.npy")
    np.save(out_file, instances_amg, allow_pickle=True)

    if _mpl_backend is not None:
        os.environ["MPLBACKEND"] = _mpl_backend
        get_active_logger().info(f"Set MPLBACKEND back to {_mpl_backend}")


setup(
    group="polarityjam",
    name="microSAM-predict",
    version="0.1.0",
    title="Segment Anything for Microscopy",
    description="A solution to create segmentations with the previously trained "
    "microSAM model fintetuned from the Segment Anything model from facebookresearch.",
    solution_creators=["Jan Philipp Albrecht"],
    tags=[
        "segmentation",
        "machine_learning",
        "images",
        "SAM",
        "facebook",
        "anything",
        "microSAM",
    ],
    license="MIT",
    documentation=["https://github.com/computational-cell-analytics/micro-sam.git"],
    covers=[],
    album_api_version="0.5.5",
    args=[
        {
            "name": "input_path",
            "type": "string",
            "required": True,
            "description": "Path (file) to the 1 channel (greyscale) image to be segmented.",
        },
        {
            "name": "output_path",
            "type": "string",
            "required": True,
            "description": "Path (directory) where the result mask will be stored.",
        },
        {
            "name": "model_name",
            "type": "string",
            "required": False,
            "description": "The name to the model to use.",
            "default": "vit_h",
        },
        {
            "name": "embedding_path",
            "type": "string",
            "required": False,
            "description": "Path (directory) where the image embedding model is stored.",
        },
        {
            "name": "checkpoint_path",
            "type": "string",
            "required": False,
            "description": "Path (file) to the checkpoint of the model to use. If none is given default url is used based on your model name.",  # noqa: E501
            "default": None,
        },
        {
            "name": "pred_iou_thresh",
            "type": "float",
            "required": False,
            "description": "The models own prediction of the masks quality. Quality is filtered by this parameter.",
            "default": 0.88,
        },
    ],
    run=run,
    dependencies={"environment_file": env_file},
)


class MicrosamSegmenter:
    """Microsam segmentation class."""

    DOWNLOAD_PATH_REL = None

    def __init__(self, params):
        """Initialize the segmenter with the given parameters."""
        self.params = params
        self.model_name = params.model_name  # type: ignore
        self.checkpoint_path = params.checkpoint_path  # type: ignore
        self.embedding_path = params.embedding_path  # type: ignore
        self.pred_iou_thresh = params.pred_iou_thresh  # type: ignore

        self.tmp_dir = self._get_tmp_dir()

        if self.checkpoint_path == "":
            self.checkpoint_path = None

        if self.embedding_path == "":
            self.embedding_path = self.tmp_dir.name

    @staticmethod
    def _get_tmp_dir():
        """Get a temporary directory."""
        import tempfile

        return tempfile.TemporaryDirectory(dir=tempfile.gettempdir())

    def segment(
        self,
        img,
        path=None,
        mode=None,
    ):
        """Segment the given image.

        Args:
            img:
                The image to segment.
            path:
                The path to the image.
            mode:
                The mode to use for segmentation. Either nucleus, organelle or cell.

        Returns:
            The segmented image as numpy array.

        """
        import os
        import tempfile

        import numpy as np
        import tifffile

        from polarityjam.controller.segmenter import SegmentationMode
        from polarityjam.polarityjam_logging import get_logger

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

        if mode == SegmentationMode.NUCLEUS:
            img = img[:, :, 1]
        elif mode == SegmentationMode.CELL:
            img = img[:, :, 0]
        elif mode == SegmentationMode.ORGANELLE:
            img = img[:, :, 2]
        elif mode == SegmentationMode.JUNCTION:
            raise ValueError("This segmentation algorithm does not support this mode!")
        else:
            raise ValueError(
                'Mode must be either "nucleus", "organelle", "cell" or "junction".'
            )

        def _install():
            # installs this file (the solution)
            path = os.path.abspath(__file__)
            album.install(path)

        solution_id = "polarityjam:microSAM-predict:0.1.0"
        album = setup_album()

        try:
            r = album.is_installed(solution_id)
            if not r:
                _install()
        except LookupError:
            _install()

        # save img to temporary folder
        self.tmp_dir = tempfile.TemporaryDirectory(dir=tempfile.gettempdir())
        img_path = os.path.join(self.tmp_dir.name, "segmentation.tif")
        tifffile.imwrite(img_path, img)

        # build argv
        argv = [
            os.path.dirname(os.path.realpath(__file__)),
            "--input_path=%s" % str(img_path),
            "--output_path=%s" % str(self.tmp_dir.name),
            "--model_name=%s" % str(self.model_name),
            "--pred_iou_thresh=%s" % str(self.pred_iou_thresh),
        ]

        if self.embedding_path is not None:
            argv.append("--embedding_path=%s" % str(self.embedding_path))

        if self.checkpoint_path is not None:
            argv.append("--checkpoint_path=%s" % str(self.checkpoint_path))

        # call microSAM solution
        album.run(solution_to_resolve=solution_id, argv=argv)

        # load segmentation
        masks = np.load(os.path.join(self.tmp_dir.name, "mask.npy"), allow_pickle=True)

        return masks

    def prepare(self, img, img_parameter):
        """Prepare the image for segmentation.

        Args:
            img:
                The image to prepare. Assumes channel last!
            img_parameter:
                The image parameter.

        Returns:
            Tuple of the prepared image and the image parameter.

        """
        import numpy as np

        from polarityjam.model.parameter import ImageParameter
        from polarityjam.polarityjam_logging import get_logger

        params_prep_img = ImageParameter()
        px_to_m_r = img_parameter.pixel_to_micron_ratio
        params_prep_img.reset()
        params_prep_img.pixel_to_micron_ratio = px_to_m_r

        numpy_img = np.zeros([img.shape[0], img.shape[1], 3])

        # check which channel is configured to use for cell segmentation:
        channel_cell_segmentation = img_parameter.channel_junction
        if self.params.channel_cell_segmentation != "":
            try:
                channel_cell_segmentation = img_parameter.__getattribute__(
                    self.params.channel_cell_segmentation
                )
            except AttributeError as e:
                raise AttributeError(
                    "Channel %s does not exist! Wrong segmentation configuration!"
                    % self.params.channel_cell_segmentation
                ) from e

        # check which channel is configured to use for nuclei segmentation:
        channel_nuclei_segmentation = img_parameter.channel_nucleus
        if self.params.channel_nuclei_segmentation != "":
            try:
                channel_nuclei_segmentation = img_parameter.__getattribute__(
                    self.params.channel_nuclei_segmentation
                )
            except AttributeError as e:
                raise AttributeError(
                    "Channel %s does not exist! Wrong segmentation configuration!"
                    % self.params.channel_nuclei_segmentation
                ) from e

        # check which channel is configured to use for organelle segmentation:
        channel_organelle_segmentation = img_parameter.channel_organelle
        if self.params.channel_organelle_segmentation != "":
            try:
                channel_organelle_segmentation = img_parameter.__getattribute__(
                    self.params.channel_organelle_segmentation
                )
            except AttributeError as e:
                raise AttributeError(
                    "Channel %s does not exist! Wrong segmentation configuration!"
                    % self.params.channel_organelle_segmentation
                ) from e

        if channel_cell_segmentation < 0:
            raise ValueError("No channel for segmentation found.")
        else:
            get_logger().info(
                "Channel used for cell segmentation at position: %s"
                % str(channel_cell_segmentation)
            )
            im_junction = img[:, :, channel_cell_segmentation]
            params_prep_img.channel_junction = (
                0  # might be called junction channel, but content depends on config
            )

            numpy_img[:, :, 0] = im_junction

        if channel_nuclei_segmentation >= 0:
            get_logger().info(
                "Channel used for nuclei segmentation at position: %s"
                % str(channel_nuclei_segmentation)
            )
            im_nucleus = img[:, :, channel_nuclei_segmentation]
            params_prep_img.channel_nucleus = (
                1  # might be called nuclei channel, but content depends on config
            )

            numpy_img[:, :, 1] = im_nucleus

        if channel_organelle_segmentation >= 0:
            get_logger().info(
                "Channel used for segmentation at position: %s"
                % str(channel_organelle_segmentation)
            )
            im_organelle = img[:, :, channel_organelle_segmentation]
            params_prep_img.channel_organelle = (
                2  # might be called organelle channel, but content depends on config
            )

            numpy_img[:, :, 2] = im_organelle

        return numpy_img, params_prep_img

    def __del__(self):
        """Clean up the temporary directory."""
        if self.tmp_dir is not None:
            self.tmp_dir.cleanup()
