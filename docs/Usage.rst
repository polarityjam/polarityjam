Usage
=====



Run options
-----------
To start the extraction feature extraction process make sure you followed the manual installation
procedure. Then run polarityjam on the comandline to look at the available run modes.
There are 3 options to start a feature extraction process run, run_stack, and run_key which
are summarized in the table below.

+------------+--------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+
| mode       | arguments                                                                | description                                                                                                                                                |
+============+==========================================================================+============================================================================================================================================================+
| run        | - paramfile.yml                                                          | Should be used when a single image needs to be processed.                                                                                                  |
|            | - input.tif                                                              |                                                                                                                                                            |
|            | - outputpath                                                             |                                                                                                                                                            |
+------------+--------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+
| run_stack  | - paramfile.yml                                                          | Should be used when a set of images in a folder needs to be processed                                                                                      |
|            | - inputpath                                                              |                                                                                                                                                            |
|            | - outputpath                                                             |                                                                                                                                                            |
+------------+--------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+
| run_key    | - paramfile.yml                                                          | Should be used when the images that needs to be processed have a complex folder structure with many subfolders that need to be excluded from the analysis  |
|            | - inputpath                                                              |                                                                                                                                                            |
|            | - inputkey.csv                                                           |                                                                                                                                                            |
|            | - outputpath                                                             |                                                                                                                                                            |
+------------+--------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+


Parameter file
--------------

Most important argument to provide for all modes is the `parmeter.yml` file. In this file in the yml format all options
can be specified how the feature extraction pipeline shout treat the data and what extraction steps to perform.
The following tables lists all options that are available for executing the pipeline. Although they are separated in
four different topics, they can be defined in a single parameter.yml.


Image Parameter
+++++++++++++++

+----------------------------+---------------+-------------------------+----------+-------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| parameter                  | Category      | type                    | default  | options     | description                                                                                                                                                      |
+============================+===============+=========================+==========+=============+==================================================================================================================================================================+
+----------------------------+---------------+-------------------------+----------+-------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| channel_junction           | image         | integer                 |          | -1,0,1,2    | Specifies which channel in the input image(s) holds information about the junction signals. -1 to indicate there is no channel.                                  |
+----------------------------+---------------+-------------------------+----------+-------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| channel_nucleus            | image         | integer                 |          | -1,0,1,2    | Specifies which channel in the input image(s) holds information about the nucleus. -1 to indicate there is no channel.                                           |
+----------------------------+---------------+-------------------------+----------+-------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| channel_organelle          | image         | integer                 |          | -1,0,1,2    | Specifies which channel in the input image(s) holds information about the organelle (e.g golgi apparatus). -1 to indicate there is no channel.                   |
+----------------------------+---------------+-------------------------+----------+-------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| channel_expression_marker  | image         | integer                 |          | -1,0,1,2    | Specifies which channel in the input image(s) holds information about the expression marker. -1 to indicate there is no channel.                                 |
+----------------------------+---------------+-------------------------+----------+-------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------+


Segmentation Parameter
++++++++++++++++++++++

+--------------------------+---------------+-------------------------+----------+-------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| parameter                | Category      | type                    | default  | options     | description                                                                                                                                                                                                |
+==========================+===============+=========================+==========+=============+============================================================================================================================================================================================================+
+--------------------------+---------------+-------------------------+----------+-------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| manually_annotated_mask  | segmentation  | string                  |          |             | Polarityjam looks for an available segmentation in the input path. This parameter specifies the suffix for manually annotated masks. Usually "_seg.npy" (cellpose default). Leave empty to use default.    |
+--------------------------+---------------+-------------------------+----------+-------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| store_segmentation       | segmentation  | bool                    | FALSE    |             | If true, stores the cellpose segmentation masks in the input path (CAUTION: not in the output path!).                                                                                                      |
+--------------------------+---------------+-------------------------+----------+-------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| use_given_mask           | segmentation  | bool                    | TRUE     |             | Indicated whether to use the masks in the input path (if any) or not. Default is true.                                                                                                                     |
+--------------------------+---------------+-------------------------+----------+-------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| model_type               | segmentation  | “custom", <model type>  | “cyto"   |             | The model type supported by your segmentation algorithm. For cellpose "cyto"  "cyto2", "custom" is possible. If "custom" is chosen, "cp_model_path" must be set.                                           |
+--------------------------+---------------+-------------------------+----------+-------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| model_path               | segmentation  | string                  |          |             | The Path to the custom model for your segmentation algorithm. Only works in combination with "cp_model_type".                                                                                              |
+--------------------------+---------------+-------------------------+----------+-------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| estimated_cell_diameter  | segmentation  | integer                 | 100      |             | The estimated cell diameter of the cells in your input image(s). Default 100.                                                                                                                              |
+--------------------------+---------------+-------------------------+----------+-------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| flow_threshold           | segmentation  | float                   | 0.4      |             | Increase this threshold if cellpose is not returning as many ROIs as you would expect. Similarly, decrease this threshold if cellpose is returning too many ill-shaped ROIs.                               |
+--------------------------+---------------+-------------------------+----------+-------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| use_gpu                  | segmentation  | bool                    | FALSE    |             | Indicates whether to use the GPU for faster segmentation. Default is false                                                                                                                                 |
+--------------------------+---------------+-------------------------+----------+-------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| clear_border             | segmentation  | bool                    | TRUE     |             | If true, removes any segmentation that is not complete because the cell protrude beyond the edge of the image.                                                                                             |
+--------------------------+---------------+-------------------------+----------+-------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| min_cell_size            | segmentation  | integer                 | 50       |             | Minimal expected cell size in pixel. Threshold value for the analysis. Cells with a smaller value will be excluded from the analysis.                                                                      |
+--------------------------+---------------+-------------------------+----------+-------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+


Runtime Parameter
+++++++++++++++++


+----------------------------+---------------+-------------------------+----------+-------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| parameter                  | Category      | type                    | default  | options     | description                                                                                                                                                      |
+============================+===============+=========================+==========+=============+==================================================================================================================================================================+
+----------------------------+---------------+-------------------------+----------+-------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| membrane_thickness         | input         | integer                 | 5        |             | Expected membrane thickness.                                                                                                                                     |
+----------------------------+---------------+-------------------------+----------+-------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| feature_of_interest        | input         | string                  | “area”   |             | Name of the feature for which a neighborhood statistics should be calculated. Any feature can be used here. Look at the features to see all available options.   |
+----------------------------+---------------+-------------------------+----------+-------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| min_cell_size              | input         | integer                 | 50       |             | Minimal expected cell size in pixel. Threshold value for the analysis. Cells with a smaller value will be excluded from the analysis.                            |
+----------------------------+---------------+-------------------------+----------+-------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| min_nucleus_size           | input         | integer                 | 10       |             | The minimal diameter of the nucleus size. Threshold value for the analysis. Cells with a nucleus with a smaller value will be excluded from the analysis.        |
+----------------------------+---------------+-------------------------+----------+-------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| min_organelle_size         | input         | integer                 | 10       |             | The minimal diameter of the organelle. Threshold value for the analysis. Cells with an organelle with a smaller value will be excluded from the analysis.        |
+----------------------------+---------------+-------------------------+----------+-------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| dp_epsilon                 | input         | integer                 | 5        |             |                                                                                                                                                                  |
+----------------------------+---------------+-------------------------+----------+-------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------+


Plot Parameter
++++++++++++++

+--------------------------+-----------+----------+---------------+-----------------------+-------------------------------------------------------------------------------------------+
| parameter                | Category  | type     | default       | options               | description                                                                               |
+==========================+===========+==========+===============+=======================+===========================================================================================+
| plot_junctions           | plot      | bool     |               |                       | Indicates whether to perform the junction polarity plot.                                  |
+--------------------------+-----------+----------+---------------+-----------------------+-------------------------------------------------------------------------------------------+
| plot_polarity            | plot      | bool     |               |                       | Indicates whether to perform the organelle polarity plot.                                 |
+--------------------------+-----------+----------+---------------+-----------------------+-------------------------------------------------------------------------------------------+
| plot_orientation         | plot      | bool     |               |                       |                                                                                           |
+--------------------------+-----------+----------+---------------+-----------------------+-------------------------------------------------------------------------------------------+
| plot_marker              | plot      | bool     |               |                       | Indicates whether to perform the marker polarity plot.                                    |
+--------------------------+-----------+----------+---------------+-----------------------+-------------------------------------------------------------------------------------------+
| plot_ratio_method        | plot      | bool     |               |                       | Indicates whether to perform the ratio plot.                                              |
+--------------------------+-----------+----------+---------------+-----------------------+-------------------------------------------------------------------------------------------+
| plot_cyclic_orientation  | plot      | bool     |               |                       | Indicates whether to perform the cyclic orientation plot.                                 |
+--------------------------+-----------+----------+---------------+-----------------------+-------------------------------------------------------------------------------------------+
| outline_width            | plot      | integer  |               |                       | Outline width of a cell. Default 2.                                                       |
+--------------------------+-----------+----------+---------------+-----------------------+-------------------------------------------------------------------------------------------+
| show_polarity_angles     | plot      | bool     | TRUE          |                       | Indicates whether to additionally add the polarity angles to the polarity plots.          |
+--------------------------+-----------+----------+---------------+-----------------------+-------------------------------------------------------------------------------------------+
| show_graphics_axis       | plot      | bool     | FALSE         |                       |                                                                                           |
+--------------------------+-----------+----------+---------------+-----------------------+-------------------------------------------------------------------------------------------+
| pixel_to_micron_ratio    | plot      | bool     |               |                       |                                                                                           |
+--------------------------+-----------+----------+---------------+-----------------------+-------------------------------------------------------------------------------------------+
| plot_scalebar            | plot      | bool     |               |                       |                                                                                           |
+--------------------------+-----------+----------+---------------+-----------------------+-------------------------------------------------------------------------------------------+
| length_scalebar_microns  | plot      | float    | 10            |                       |                                                                                           |
+--------------------------+-----------+----------+---------------+-----------------------+-------------------------------------------------------------------------------------------+
| graphics_output_format   | plot      | string   | “png”, “pdf”  | “png”, “pdf” , “svg"  | The output format of the plot figures. Several can be specified. Default is png and pdf.  |
+--------------------------+-----------+----------+---------------+-----------------------+-------------------------------------------------------------------------------------------+
| dpi                      | plot      | integer  | 300           |                       | Resolution of the plots. Specifies the dots per inch. Default is 300.                     |
+--------------------------+-----------+----------+---------------+-----------------------+-------------------------------------------------------------------------------------------+
| graphics_width           | plot      | integer  | 5             |                       | The width of the output plot figures in inches. Default 5.                                |
+--------------------------+-----------+----------+---------------+-----------------------+-------------------------------------------------------------------------------------------+
| graphics_height          | plot      | integer  | 5             |                       | The width of the output plot figures in inches. Default 5.                                |
+--------------------------+-----------+----------+---------------+-----------------------+-------------------------------------------------------------------------------------------+



Key file
--------

Often, analysts are challenged not only with the problem of actually performing the analysis,
but also how and where to store the data. Iterative acquisition of images as well as various
experimental settings sometimes require complex folder structures and naming schema to organize data.
Frequently, researchers phase the problem of data being distributed over several physical devices,
leaving them with the problem of how to execute a certain tool on a dedicated subset of images.
Not often a high amount of time is necessary before the analysis is finally performed.
Moreover, performing analysis steps on several experimental conditions often requires repeating the
whole pipeline several times to get the desired output. To tackle this problem,
polarityjam offers the execution option run_key that accepts a csv file describing the storage
structures and conditions. To still be able to migrate the data without altering the csv,
paths are relative to a given root folder (e.g. inputpath).

The structure of the csv is given as follows:


+--------------+-------------+
| folder_name  | short_name  |
+==============+=============+
| set_1        | cond_1      |
+--------------+-------------+
| set_2        | cond_2      |
+--------------+-------------+


Folder structure will also be created in the provided output path. Specify a short_name different to the folder_name to rename each folder. (e.g. folder set_1 will be named cond_1 in the output path)

.. warning::
    Using OS specific paths here might hurt reproducibility! (e.g. windows paths are different than unix paths!)

Webb app
--------

The R-shiny web app further analyses the results of the feature extraction process in the browser.
There are several statistics available which parameters can be adapted during runtime to immediately
observe the change in the corresponding visualization. Thus, exploring the data and revealing
interesting patterns is heavily facilitated. To get to know more about the statics jump to circular
statistics and continue reading or visit the method section.


Testing
-------

We use a testing framework to make sure outcomes are as expected. To run the software with our example data provided
in the package use the following command:

.. code-block:: console

    polarityjam_test

This will not keep the output on the disk. To look at the output of the tests specify a target folder:

.. code-block:: console

    polarityjam_test --target-folder=/tmp/mytarget



