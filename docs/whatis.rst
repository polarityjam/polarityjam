What is PolarityJaM?
====================

PolarityJaM suite
-----------------

The PolarityJaM suite is a software package that lets you study cell polarity, junctions and morphology.

.. image:: images/sketch-data-analysis-organization-v5_.png

At the heart lies your data that you wish to explore. That usually involves three steps: (1) you extract relevant
features from your data, (2) you analyze the features and (3) you visualize your results. Moreover, you want your work
to be reproducible and easy to understand and eventually extend by others. PolarityJaM is designed to help you with
all of these tasks.

The analysis of images performed by PolarityJaM is divided into two individual processing parts that
can be executed individually from each other. First, all necessary information from the input is
extracted via a feature extraction pipeline. The output will be a csv file that can be easily
shared and interpreted. In an independent second step these features can be then statistically
analysed in the web-app and interesting patterns can be reviled. Usually, exploring your data takes
a high amount of time and domain knowledge.
To minimize the amount of time needed to quickly look at complex cell patterns and potentially
discover interesting collective behavior, visualizing the features can be done interactively in an
online setting. Simply visit `www.polarityjam.com <www.polarityjam.com>`_ to get a first feeling of what is possible
without the need to execute anything.


Feature Extraction Pipeline
---------------------------
The feature extraction pipeline lets you extract relevant features from your image(s) and collect
them in a csv file. Optionally, the user can choose between several options to visualize the input
images(s) and have a first look at the result of the information extraction process. If you want to
know what features can be extracted visit the `Usage <Usage>`_ section.
If you want to know more about the features, visualizations and methodology behind
this process check out the method section.

Often, an analysis performed is barely reproducible due to missing information about version,
specific parameters and more. To be able to repeat the extraction process at a later point in time
the PolarityJaM feature extraction pipeline additionally stores all necessary meta-information.
This includes both, the full log output of the extraction process, as well as the specific parameters
the pipeline was started with. This allows to perform several analysis on different data and continue
or repeat previous feature extraction processes without spending much effort into organizing the meta
information. Additionally, the informative logging system supports you with information about the
current process.


Online Service
--------------

Once the feature extraction process is finished, the information can be statistically analysed to
discover structural patterns. Easiest way to achieve this is to use the service hosted at
`www.polarityjam.com <www.polarityjam.com>`_ by simply uploading the csv from the feature
extraction process. Alternatively, the app can also be executed locally.