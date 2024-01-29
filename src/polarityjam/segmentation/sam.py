"""Module holding the SAM segmentation solution."""
from album.runner.api import get_args, setup

# DO NOT IMPORT ANYTHING OTHER THAN FROM RUNNER API HERE

env_file = """name:  SAM
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.9
  - tifffile>=2023.7.18
  - pip
  - pip:
    - segment-anything
    - torch
    - torchvision
    - torchaudio
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

    import os

    import numpy as np
    import tifffile
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    # open tiff
    img_3d = tifffile.TiffFile(args.input_path).asarray()

    sam = sam_model_registry[args.model_name](checkpoint=str(args.model_path))
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(img_3d)

    # save masks
    out_file = os.path.join(args.output_path, "mask.npy")
    np.save(out_file, masks, allow_pickle=True)


setup(
    group="polarityjam",
    name="SAM-predict",
    version="0.1.0",
    title="Segment Anything Segmentation",
    description="A solution to create segmentations with the previously trained "
    "Segment Anything model from facebookresearch.",
    solution_creators=["Jan Philipp Albrecht"],
    tags=[
        "segmentation",
        "machine_learning",
        "images",
        "SAM",
        "facebook",
        "2D",
        "anything",
    ],
    license="Apache License 2.0",
    documentation=["https://github.com/facebookresearch/segment-anything"],
    covers=[],
    album_api_version="0.5.5",
    args=[
        {
            "name": "input_path",
            "type": "string",
            "required": True,
            "description": "Path (file) to the 3 channel (RGB) image to be segmented.",
        },
        {
            "name": "model_path",
            "type": "string",
            "required": True,
            "description": "Path (directory) where the segmentation model is stored.",
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
    ],
    run=run,
    dependencies={"environment_file": env_file},
)


class SamSegmenter:
    """SAM segmentation class."""

    DOWNLOAD_PATH_REL = None

    def __init__(self, params):
        """Initialize the segmenter with the given parameters."""
        import os
        from pathlib import Path

        self.params = params
        self.DOWNLOAD_PATH_REL = Path(
            os.path.dirname(os.path.realpath(__file__))
        ).joinpath("SAM")
        self.model_url = params.model_url  # type: ignore
        self.model_name = params.model_name  # type: ignore
        self.model_path = self.DOWNLOAD_PATH_REL.joinpath(self.model_name)
        self.tmp_dir = None

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

        solution_id = "polarityjam:SAM-predict:0.1.0"
        album = setup_album()

        try:
            r = album.is_installed(solution_id)
            if not r:
                _install()
        except LookupError:
            _install()

        # prepare image for SAM
        img_e = np.expand_dims(img, axis=-1)
        img_3d = np.repeat(img_e, 3, axis=-1).astype(np.uint8)

        # save img to temporary folder
        self.tmp_dir = tempfile.TemporaryDirectory(dir=tempfile.gettempdir())
        img_path = os.path.join(self.tmp_dir.name, "segmentation.tif")
        tifffile.imwrite(img_path, img_3d)

        # build argv
        argv = [
            os.path.dirname(os.path.realpath(__file__)),
            "--input_path=%s" % str(img_path),
            "--model_path=%s" % str(self.model_path),
            "--output_path=%s" % str(self.tmp_dir.name),
        ]

        # call SAM solution
        album.run(solution_to_resolve=solution_id, argv=argv)

        # load segmentation
        masks = np.load(os.path.join(self.tmp_dir.name, "mask.npy"), allow_pickle=True)

        return self._to_instance_segmentation(masks)

    @staticmethod
    def _to_instance_segmentation(masks):
        import numpy as np

        h = np.zeros(masks[0]["segmentation"].shape)

        for idx, i in enumerate(masks):
            if idx == 0:
                continue  # background class

            f = i["segmentation"].astype(np.uint8)
            f = f * idx
            h = np.where(f > 0, f, h)

        return h.astype(np.uint8)

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
        from polarityjam.utils.url import download_resource

        if not self.model_path.exists():
            if not self.model_url:
                raise ValueError(
                    "Model does not exist and no model url is given. Please specify a model url in the parameters."
                )
            download_resource(self.model_url, self.DOWNLOAD_PATH_REL)

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
                1  # might be called nucleus channel, but content depends on config
            )

            numpy_img[:, :, 1] = im_nucleus

        if channel_organelle_segmentation >= 0:
            get_logger().info(
                "Channel used for organelle segmentation at position: %s"
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
