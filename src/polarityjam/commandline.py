"""Commandline entrypoint for PolarityJam."""
import json
import os
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from polarityjam import load_segmenter
from polarityjam.controller.extractor import Extractor
from polarityjam.controller.plotter import Plotter
from polarityjam.controller.segmenter import SegmentationMode
from polarityjam.model.collection import PropertiesCollection
from polarityjam.model.masks import BioMedicalJunctionSegmentation
from polarityjam.model.parameter import (
    ImageParameter,
    PlotParameter,
    RuntimeParameter,
    read_parameters,
)
from polarityjam.polarityjam_logging import get_doc_file_prefix, get_logger
from polarityjam.utils.io import (
    create_path_recursively,
    get_tif_list,
    read_image,
    read_key_file,
    write_dict_to_yml,
)


def run(args):
    """Run the polarityjam pipeline.

    Args:
        args:
            The arguments provided by the user.

    """
    # read args
    param_file = args.param
    filepath = args.in_file
    filename = args.filename_prefix
    output_path = args.out_path

    # print info
    get_logger().info(
        "Arguments provided: %s"
        % json.dumps(
            {
                "param": param_file,
                "in_file": filepath,
                "out_path": output_path,
                "filename": filename,
            },
            sort_keys=True,
            indent=4,
        )
    )

    # process args
    if not filename:
        filename, _ = os.path.splitext(os.path.basename(filepath))

    parameters = read_parameters(param_file)
    get_logger().info(
        "Parameters: %s" % json.dumps(parameters, sort_keys=True, indent=4)
    )

    # start routine
    _run(filepath, parameters, output_path, filename)
    _finish(parameters, output_path)


def _finish(parameters, output_path):
    # write parameters to disk
    out_param = Path(output_path).joinpath(
        "{}_param{}".format(get_doc_file_prefix(), ".yml")  # noqa: P101
    )
    get_logger().info("Writing parameter to disk: %s" % out_param)
    write_dict_to_yml(out_param, parameters)


def _run(
    infile: Union[Path, str],
    param: dict,
    output_path: Union[Path, str],
    fileout_name: str,
):
    create_path_recursively(output_path)

    # read input
    img = read_image(infile)
    params_img = ImageParameter(param)

    # inputParams
    params_input = RuntimeParameter(param)

    # plotter
    params_plot = PlotParameter(param)
    p = Plotter(params_plot)

    # segmenter
    s, _ = load_segmenter(params_input, param)

    # prepare segmentation and plot
    img_seg, img_seg_params = s.prepare(img, params_img)
    p.plot_channels(img_seg, img_seg_params, output_path, fileout_name, close=True)

    # segment
    mask = s.segment(img_seg, infile)

    mask_nuclei = None
    if img_seg_params.channel_nucleus is not None:
        mask_nuclei = s.segment(
            img_seg[img_seg_params.channel_nucleus],
            infile,
            mode=SegmentationMode.NUCLEUS,
        )

    mask_junction = None
    if img_seg_params.channel_junction is not None:
        infile_path = Path(infile)
        input_mask_junc = infile_path.parent.joinpath(
            infile_path.stem + "_seg_edge.npy"
        )
        if input_mask_junc.exists():
            get_logger().info("Loading junction mask from %s" % str(input_mask_junc))
            mask_junc_dict = np.load(str(input_mask_junc), allow_pickle=True)
            mask_junction = BioMedicalJunctionSegmentation(
                mask_junc_dict["edge_masks"],
                mask_junc_dict["edge_ids"],
                mask_junc_dict["edge_cell_ids"],
            )

    # plot cellpose mask
    p.plot_mask(
        mask,
        img_seg,
        img_seg_params,
        output_path,
        fileout_name,
        mask_nuclei,
        close=True,
    )

    # feature extraction
    c = PropertiesCollection()
    e = Extractor(params_input)
    e.extract(
        img,
        params_img,
        mask,
        fileout_name,
        output_path,
        c,
        segmentation_mask_nuclei=mask_nuclei,
        segmentation_mask_junction=mask_junction,
    )

    # visualize
    p.plot_collection(c, close=True)

    get_logger().info("Head of created dataset: \n %s" % c.dataset.head())

    # write output
    fileout_base, _ = os.path.splitext(fileout_name)
    fileout_path = Path(output_path).joinpath(fileout_base + ".csv")
    collection_out_path = Path(output_path).joinpath(fileout_base + ".pkl")
    get_logger().info("Writing collection to disk: %s" % fileout_path)
    c.save(collection_out_path)
    get_logger().info("Writing features to disk: %s" % fileout_path)
    c.dataset.to_csv(str(fileout_path), index=False)

    get_logger().info("File %s done!" % infile)

    return c.dataset, mask


def run_stack(args):
    """Run the polarityjam pipeline on a stack of images.

    Args:
        args:
            The arguments provided by the user.

    """
    # read args
    param_file = args.param
    inpath = args.in_path
    output_path = args.out_path

    # print info
    get_logger().info(
        "Arguments provided: %s"
        % json.dumps(
            {"param": param_file, "in_path": inpath, "out_path": output_path},
            sort_keys=True,
            indent=4,
        )
    )

    # process
    parameters = read_parameters(param_file)
    get_logger().info(
        "Parameters: %s" % json.dumps(parameters, sort_keys=True, indent=4)
    )

    file_list = get_tif_list(inpath)
    merged_properties_df = pd.DataFrame()

    for filepath in file_list:
        filepath = Path(filepath)
        filename = filepath.stem + filepath.suffix

        if not ((filepath.suffix != ".tif") or (filepath.suffix != ".tiff")):
            continue

        get_logger().info(
            'Processing file with file stem  "%s" and file extension: "%s"'
            % (filepath.stem, filepath.suffix)
        )

        # single run
        properties_df, cellpose_mask = _run(filepath, parameters, output_path, filename)

        if merged_properties_df.empty:
            merged_properties_df = properties_df.copy()
        else:
            merged_properties_df = pd.concat(
                [merged_properties_df, properties_df], ignore_index=True
            )

        merged_file = str(Path(output_path).joinpath("merged_properties.csv"))
        get_logger().info("Writing merged features to disk: %s" % merged_file)
        merged_properties_df.to_csv(merged_file, index=False)

    _finish(parameters, output_path)


def run_key(args):
    """Run the polarityjam pipeline on a stack of images given an additional csv describing the folder structure.

    Args:
        args:
            The arguments provided by the user.

    """
    # read args
    param_file = args.param
    in_path = args.in_path
    inkey = args.in_key
    output_path_base = args.out_path

    # print info
    get_logger().info(
        "Arguments provided: %s"
        % json.dumps(
            {"param": param_file, "in_key": inkey, "out_path": output_path_base},
            sort_keys=True,
            indent=4,
        )
    )

    # convert
    output_path_base = Path(output_path_base)
    create_path_recursively(output_path_base)
    in_path = Path(in_path)

    # process
    parameters = read_parameters(param_file)
    get_logger().info(
        "Parameters: %s" % json.dumps(parameters, sort_keys=True, indent=4)
    )
    key_file = read_key_file(inkey)

    condition_cols = parameters["keyfile_condition_cols"]
    if len(condition_cols) == 0:
        raise ValueError(
            "No condition columns specified in parameter file! "
            "Please set 'keyfile_condition_cols' in the parameter file."
        )
    unique_condition_identifier = condition_cols[0]

    # empty DF summarizing overall results
    summary_df = pd.DataFrame()
    summary_properties_df = pd.DataFrame()

    offset = 0
    for _, row in key_file.iterrows():
        # current stack input sub folder
        cur_sub_path = str(row["folder_name"])
        if cur_sub_path.startswith(os.path.sep):
            cur_sub_path = cur_sub_path[1:-1]
        input_path = in_path.joinpath(cur_sub_path)

        # current stack output sub-folder
        cur_sub_out_path = str(row[unique_condition_identifier])
        output_path = output_path_base.joinpath(cur_sub_out_path)

        # empty results dataset for each condition
        merged_properties_df = pd.DataFrame()

        file_list = get_tif_list(input_path)
        get_logger().info("Search for images in folder: %s" % (str(input_path)))
        get_logger().info("Image list: %s" % file_list)
        for file_index, filepath in enumerate(file_list):
            filepath = Path(filepath)
            filename = filepath.stem + filepath.suffix

            if not ((filepath.suffix != ".tif") or (filepath.suffix != ".tiff")):
                continue

            get_logger().info(
                "Processing file with: file stem  %s and file extension: %s"
                % (filepath.stem, filepath.suffix)
            )

            # single run
            properties_df, cellpose_mask = _run(
                filepath, parameters, output_path, filename
            )

            # append condition
            for condition_col in parameters["keyfile_condition_cols"]:
                properties_df[condition_col] = row[condition_col]

            if merged_properties_df.empty:
                merged_properties_df = properties_df.copy()
            else:
                merged_properties_df = pd.concat(
                    [merged_properties_df, properties_df], ignore_index=True
                )

            merged_file = str(
                output_path.joinpath(
                    "merged_table_%s" % row[unique_condition_identifier] + ".csv"
                )
            )
            get_logger().info("Writing merged features to disk: %s" % merged_file)
            merged_properties_df.to_csv(merged_file, index=False)

            summary_df.at[offset + file_index, "folder_name"] = row["folder_name"]
            for condition_col in parameters["keyfile_condition_cols"]:
                summary_df.at[offset + file_index, condition_col] = row[condition_col]
            summary_df.at[offset + file_index, "filepath"] = filepath
            summary_df.at[offset + file_index, "cell_number"] = len(
                np.unique(cellpose_mask)
            )

        offset = offset + len(file_list)

        summary_df_path = output_path_base.joinpath("summary_table" + ".csv")
        get_logger().info("Writing summary table to disk: %s" % summary_df_path)
        summary_df.to_csv(str(summary_df_path), index=False)

        if summary_properties_df.empty:
            summary_properties_df = merged_properties_df.copy()
        else:
            summary_properties_df = pd.concat(
                [summary_properties_df, merged_properties_df], ignore_index=True
            )

        summary_file = str(output_path_base.joinpath("summary_table_properties.csv"))
        get_logger().info("Writing merged features to disk: %s" % summary_file)
        summary_properties_df.to_csv(summary_file, index=False)

        keyfile_path = output_path_base.joinpath("key_file" + ".csv")
        get_logger().info("Writing key file to disk: %s" % keyfile_path)
        key_file.to_csv(str(keyfile_path), index=False)

    _finish(parameters, output_path_base)
