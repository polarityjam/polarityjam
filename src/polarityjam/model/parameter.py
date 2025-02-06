"""Parameter classes holding the configuration of the feature extraction process."""
import os
from importlib import import_module
from pathlib import Path

import yaml
from aenum import Enum, extend_enum

from polarityjam.settings import Settings
from polarityjam.utils.io import get_dict_from_yml, list_files_recursively


def _obj_constructor(cls, *args, **kwargs):
    """Convert static attributes from class to instance attributes and sets user values."""
    self = object.__new__(cls)

    # convert static attributes to instance attributes
    static_attrs = [
        v for v, m in vars(cls).items() if not (v.startswith("_") or callable(m))
    ]
    for attr in static_attrs:
        setattr(self, attr, getattr(cls, attr))

    if args != ():
        for dictionary in args:
            for key in dictionary:
                self._setattr(key, dictionary[key])

    if kwargs != {}:
        for key in kwargs:
            self._setattr(key, kwargs[key])

    return self


class Parameter:
    """Base class for all parameters."""

    def __init__(self, *args, **kwargs):
        """Initialize the parameter class with its default values."""
        # init with default values from resources
        current_path = Path(os.path.dirname(os.path.realpath(__file__)))
        param_base_file = Path(current_path).joinpath(
            "..", "utils", "resources", "parameters.yml"
        )
        args_init = get_dict_from_yml(param_base_file)

        for key in args_init:
            self._setattr(key, args_init[key])

        if args != ():
            for dictionary in args:
                for key in dictionary:
                    self._setattr(key, dictionary[key])

        if kwargs != {}:
            for key in kwargs:
                self._setattr(key, kwargs[key])

    def _setattr(self, key, val):
        if hasattr(self, key):
            setattr(self, key, val)

    def reset(self):
        """Reset all parameters to None."""
        for key in self.__dict__:
            self._setattr(key, None)

    @classmethod
    def from_yml(cls, path: str):
        """Create a parameter object from a yml file."""
        params = get_dict_from_yml(Path(path))

        return cls(params)

    def __str__(self, indent=1):
        """Return a string representation of the parameter object."""
        s = "%s:  \n" % self.__class__.__name__
        for attr in self.__dict__:
            s += f"{attr:<30}{str(getattr(self, attr)):<40}\n"
        return s


class ImageParameter(Parameter):
    """Parameter class for the parameters necessary for the image processing."""

    def __init__(self, attrs=None):
        """Initialize the image parameter class."""
        if attrs is None:
            attrs = {}
        self.channel_junction = None
        self.channel_nucleus = None
        self.channel_organelle = None
        self.channel_expression_marker = None
        self.pixel_to_micron_ratio = None

        super().__init__(**attrs)


class SegmentationAlgorithmE(Enum):
    """Enum for the segmentation algorithm."""

    pass  # constructed at runtime


class SegmentationParameterE(Enum):
    """Enum for the segmentation parameters."""

    pass  # constructed at runtime


class RuntimeParameter(Parameter):
    """Parameter class for the parameters necessary for the calculation of the features."""

    def __init__(self, attrs=None):
        """Initialize the runtime parameter class."""
        if attrs is None:
            attrs = {}
        self.extract_group_features = None
        self.extract_morphology_features = None
        self.extract_polarity_features = None
        self.extract_intensity_features = None
        self.membrane_thickness = None
        self.junction_threshold = None
        self.feature_of_interest = None
        self.min_cell_size = None
        self.min_nucleus_size = None
        self.min_organelle_size = None
        self.dp_epsilon = None
        self.cue_direction = None
        self.connection_graph = None
        self.segmentation_algorithm = None
        self.remove_small_objects_size = None
        self.clear_border = None
        self.save_sc_image = None
        self.keyfile_condition_cols = None

        super().__init__(**attrs)


class SegmentationParameter(Parameter):
    """Parameter class for the parameters necessary for the segmentation of images."""

    def __new__(cls, segmentation_algorithm: str, attrs=None):
        """Create a new segmentation parameter object."""
        segmentation_parameter_type = cls.create_segmentation_parameter(
            segmentation_algorithm
        )
        if attrs is None:
            attrs = {}
        return segmentation_parameter_type(**attrs)

    def to_dict(self):
        """Convert the segmentation parameter object to a dictionary."""
        return self.__dict__

    @staticmethod
    def create_segmentation_parameter(segmentation_algorithm: str) -> type:
        """Create a segmentation parameter class for the segmentation process."""
        d = {
            # constructor
            "__new__": _obj_constructor,
            "to_dict": SegmentationParameter.to_dict,
        }
        d.update(get_dict_from_yml(SegmentationParameterE[segmentation_algorithm].value))  # type: ignore
        segmentation_parameter_type = type("SegmentationParameter", (Parameter,), d)
        return segmentation_parameter_type

    @classmethod
    def from_yml(cls, path: str):
        """Create a parameter object from a yml file."""
        params = get_dict_from_yml(Path(path))

        if "segmentation_algorithm" in params:
            segmentation_parameter_type = cls.create_segmentation_parameter(
                RuntimeParameter(params).segmentation_algorithm
            )
            return segmentation_parameter_type(params)
        else:
            segmentation_parameter_type = cls.create_segmentation_parameter(
                Settings.segmentation_algorithm_default.value
            )
            return segmentation_parameter_type(params)


class PlotParameter(Parameter):
    """Parameter class for the parameters necessary for the plotting of the results."""

    def __init__(self, attrs=None):
        """Initialize the plot parameter class."""
        if attrs is None:
            attrs = {}
        self.plot_junctions = None
        self.plot_polarity = None
        self.plot_elongation = None
        self.plot_circularity = None
        self.plot_marker = None
        self.plot_ratio_method = None
        self.plot_symmetry = None
        self.plot_shape_orientation = None
        self.plot_foi = None
        self.plot_sc_image = None
        self.plot_threshold_masks = None
        self.plot_sc_partitions = None

        self.show_statistics = None
        self.show_polarity_angles = None
        self.show_graphics_axis = None
        self.show_scalebar = None

        self.outline_width = None
        self.length_scalebar_microns = None
        self.length_unit = None

        self.graphics_output_format = None
        self.dpi = None
        self.graphics_width = None
        self.graphics_height = None
        self.membrane_thickness = None  # todo: remove me

        self.fontsize_text_annotations = None
        self.font_color = None
        self.marker_size = None
        self.alpha = None
        self.alpha_cell_outline = None

        super().__init__(**attrs)


def read_parameters(parameter_file: str) -> dict:
    """Read in default parameters and replaces user defined parameters.

    Args:
        parameter_file:
            Path to the parameter file.
    Returns:
        dictionary where all missing parameter values are replaced with the default values.

    """
    current_path = Path(os.path.dirname(os.path.realpath(__file__)))

    param_base_file = Path(current_path).parent.joinpath(
        "utils", "resources", "parameters.yml"
    )

    with open(param_base_file) as yml_f:
        parameters = yaml.safe_load(yml_f)

    with open(parameter_file) as file:
        parameters_local = yaml.safe_load(file)

    # overwrite global parameters with local setting
    for key in parameters_local:
        parameters[key] = parameters_local[key]

    # load segmentation parameters
    sp = SegmentationParameter(
        parameters[Settings.segmentation_algorithm.value], parameters_local
    )

    for k in sp.__dict__:
        parameters[k] = getattr(sp, k)

    return parameters


def _load_dynamic_segmenter():
    m = import_module(Settings.dynamic_loading_prefix.value)
    path = Path(os.path.abspath(m.__file__)).parent
    for yml_spec in list_files_recursively(path, endswith=".yml", recursive=False):
        spec_dict = get_dict_from_yml(yml_spec)
        name = spec_dict["path"].split(".")[-1]
        value = "{}.{}".format(  # noqa: P101
            Settings.dynamic_loading_prefix.value, spec_dict["path"]
        )
        extend_enum(SegmentationAlgorithmE, name, value)
        extend_enum(SegmentationParameterE, name, path.joinpath(yml_spec))


_load_dynamic_segmenter()
