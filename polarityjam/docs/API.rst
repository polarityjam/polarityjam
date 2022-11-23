Python API
==========

You can use polarityjam within python to do your analysis. A good tool to get an impression for the data you are
about to analyse is jupyter notebook (`https://jupyter.org/ <https://jupyter.org/>`_). Paired with python anaconda
it is a powerful tool.
The following will show how to use polarityjam in a python shell, or preferably in a jupyter notebook.


Images
------
    An image can be read in and needs to be further specified what channels hold which information.
    For that the `ImageParameter` class should be used.

    .. code-block:: python

        from polarityjam import ImageParameter
        from polarityjam.utils.io import read_image

        infile = "060721_EGM2_18dyn_01.tif"

        output_path = "out_folder

        img = read_image(infile)

        # describe your image with ImageParameter
        params_image = ImageParameter()

        # set the channels for the Image 060721_EGM2_18dyn_01.tif
        params_image.channel_organelle = 0
        params_image.channel_nucleus = 2
        params_image.channel_junction = 3
        params_image.channel_expression_marker = 3

        # look at the configuration
        print(params_image)

    out:

    .. code-block:: text

        ImageParameter:
        channel_junction              3
        channel_nucleus               2
        channel_organelle             0
        channel_expression_marker     3



Segmentation
------------
    Before features can be extracted the image must be segmented. Use the `CellposeSegmenter` class for it.

    .. code-block:: python

        from polarityjam import CellposeSegmenter, SegmentationParameter

        params_seg = SegmentationParameter()  # use default values. print to see the configuration.

        s = CellposeSegmenter(params_seg)

        # prepare your image for segmentation
        img_prepared, img_prepared_params = s.prepare(img, params_image)

        # segment your image
        mask = s.segment(img_prepared, infile)

Feature Extraction
------------------
    Use the `Extractor` class to extract the features from your segmentation.

    .. code-block:: python

        from polarityjam import Extractor, PropertiesCollection
        from polarityjam import RuntimeParameter

        params_runtime = RuntimeParameter()   # use default values. print to see the configuration.

        c = PropertiesCollection()
        e = Extractor(params_runtime)

        e.extract(img, params_image, mask, fileout_name, output_path, c)

        c.dataset.head()

    out:

    .. code-block:: text

        filename	label	cell_X	cell_Y	cell_shape_orientation	cell_major_axis_length	cell_minor_axis_length	cell_eccentricity	cell_major_to_minor_ratio	cell_area	...	morans_p_norm	neighbors_cell	neighbors_mean_dif_1st	neighbors_median_dif_1st	neighbors_stddev_dif_1st	neighbors_range_dif_1st	neighbors_mean_dif_2nd	neighbors_median_dif_2nd	neighbors_stddev_dif_2nd	neighbors_range_dif_2nd
        1	060721_EGM2_18dyn_01	17.0	68.477653	804.993960	2.525143	114.164720	69.789749	0.791393	1.635838	5795.0	...	0.096469	3.0	-499.333333	-265.0	855.157036	2055.0	2421.250000	2092.0	3161.303637	7365.0
        2	060721_EGM2_18dyn_01	18.0	112.365152	216.717906	1.772854	131.758840	79.042332	0.800074	1.666940	7260.0	...	0.096469	4.0	830.500000	1023.5	997.362647	2629.0	1808.666667	1028.0	3788.546540	11784.0
        3	060721_EGM2_18dyn_01	19.0	88.122335	962.746247	0.031579	118.404769	67.778780	0.819952	1.746930	5395.0	...	0.096469	1.0	-1244.000000	-1244.0	0.000000	0.0	267.500000	267.5	132.500000	265.0
        4	060721_EGM2_18dyn_01	20.0	155.107352	459.765362	1.453797	157.325719	138.722923	0.471705	1.134100	16404.0	...	0.096469	4.0	-6958.750000	-6791.0	2145.264713	5777.0	-10233.000000	-10853.0	1680.279739	5444.0
        5	060721_EGM2_18dyn_01	21.0	107.061672	906.465671	2.409574	144.501883	37.908917	0.964975	3.811818	4151.0	...	0.096469	3.0	1422.333333	1379.0	166.149197	400.0	5066.000000	5066.0	3011.000000	6022.0
        5 rows × 63 columns



Visualization
-------------
    At each point in the process visualization is necessary for quality control. Use the `Plotter` class to look at
    processing steps.

    .. code-block:: python

        from polarityjam import Plotter
        from polarityjam import PlotParameter

        params_plot = PlotParameter()

        p = Plotter(params_plot)

        # plot the channels of the image ready for segmentation
        p.plot_channels(img_prepared, img_prepared_params, output_path, infile)

        # plot the segmentation mask
        p.plot_mask(mask, img_prepared, img_prepared_params, output_path, fileout_name)

        # plot the entire collection
        p.plot_collection(c)


Building your pipeline
----------------------

By combining everything from above your own pipeline can be build. Look at the `jupyter notebook <https://github.com/wgiese/polarityjam/blob/main/polartyjam-notebook.ipynb>`_ file to see
everything in action.

