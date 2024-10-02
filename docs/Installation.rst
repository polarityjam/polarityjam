Installation
============


Polarity-JaM - Feature Extraction Pipeline
------------------------------------------

The feature extraction pipeline is the process of extracting all relevant features from all input images.
The result will always be a ``.csv`` file for each image containing its individual cells as rows and their
corresponding feature values as columns. Additionally, plots will be created.
These visualizations can be used for quality control but might also be suitable for a publication.


Manual installation
++++++++++++++++++++++++++++++++++++++++++++++++++++++++
For additional support in segmentation, we suggest to install Polarity-JaM
via micromamba and the conda-forge channel. For that, make sure you have
`micromamba <https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html>`_ installed.
Manually install Polarity-JaM via:

.. code-block:: console

    micromamba create -y -n polarityjam python=3.8 pip -c conda-forge
    micromamba activate polarityjam
    pip install polarityjam


Manual installation of the napari plugin for polarityjam
++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Make sure you have Polarity-JaM installed. Then execute the following steps:

.. code-block:: console

    micromamba activate polarityjam
    pip install polarityjam-napari-plugin

To start polarityjam with napari viewer execute:

.. code-block:: console

    polarityjam napari


Automatic installation with album
+++++++++++++++++++++++++++++++++

Install `album <https://album.solutions/>`_. This installation comes natively with
micromamba. Then execute the following steps:

.. code-block:: console

    album add-catalog https://gitlab.com/album-app/catalogs/helmholtz-imaging
    album install de.mdc-berlin:polarityjam:0.1.0

You can now run the pipeline with the following command:

.. code-block:: console

    album run de.mdc-berlin:polarityjam:0.1.0

.. note::
    Please make sure you are using album version 0.10.4 (current version) for installation of the polarityjam solution.


Manual installation from GitHub
+++++++++++++++++++++++++++++++

Make sure you have `micromamba <https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html>`_ installed.

Execute the following steps on the commandline:

.. code-block:: console

    micromamba create -y -n polarityjam python=3.8 -c conda-forge
    micromamba activate polarityjam
    git clone https://github.com/polarityjam/polarityjam.git # via git or download via browser
    cd polarityjam
    pip install -e .


Polarity-JaM - Web App
----------------------

The R-shiny Polarity-JaM web app further analyses the results of the feature extraction process in the browser.
There are several statistics available which parameters can be adapted during runtime to immediately
observe the change in the corresponding visualization.
Thus, Exploring the data and relieving interesting patterns is heavily facilitated.
To get to know more about the statics continue reading or visit the :any:`Methods <methods>` section.

.. note::
    You don't need to install the web app to use the feature extraction pipeline. The web app is
    our visualization tool for the results of the feature extraction pipeline.
    You can simply use our online service `here <http://www.polarityjam.com>`_.
    Or visit :ref:`software suite <software suite>` for more information.

Manual installation
+++++++++++++++++++

Make sure you have `micromamba <https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html>`_ installed.

Execute the following steps on the commandline:

.. code-block:: console

    git clone https://github.com/polarityjam/polarityjam-app.git # via git or download via browser
    cd polarityjam-app
    micromamba env create -f polarityjam-app.yml
    micromamba activate polarityjam-app
    cd app
    Rscript app.R

Open the browser in the URL given in the output of the R-shiny call (usually http://127.0.0.1:8888 ).


Automatic installation with album
+++++++++++++++++++++++++++++++++


Install `album <https://album.solutions/>`_. This installation comes natively with
micromamba. Then execute the following steps:

.. code-block:: console

    album add-catalog https://gitlab.com/album-app/catalogs/helmholtz-imaging
    album install de.mdc-berlin:polarityjam-app:0.1.0

You can now run the pipeline with the following command:

.. code-block:: console

    album run de.mdc-berlin:polarityjam-app:0.1.0

A browser should automatically open. If not, open http://127.0.0.1:8888

Run with Rstudio
++++++++++++++++

Alternatively, you can also open the app.R your local polarityjam-app/app folder with Rstudio
and simply click on "Run App".