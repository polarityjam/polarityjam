Web-App
========

We provide a simple means for statistical analysis of circular data that supports both exploratory analysis and the generation of graphs for publication. The App is based on the R Shiny inteface and can be started locally on you computer or on an web service provided by the Max-Delbrück-Center for MolecularMedicine.

Circular statistics
-------------------

The App provides a set of functions for the analysis and plotting of circular data. Generally circular data are data
that can be mapped on a circlular scale, such as direction or time of day. Unlike linear data, which is measured on a
straight line scale, circular data has a natural periodicity, meaning that it repeats itself after a certain point.
Besides cellular polarity, examples outside the biology are the direction of the wind,
the direction of the magnetic field.

Here, we generally distinguish between directional, axial and linear data.
Directional data are data with values in [0,2 &pi ] or 0 to 360, in radians or degrees, respectively.
Axial data are data with values [0, &pi] or 0 to 180 in degrees, meaning that it repeats itself after 180 degrees.
Linear data is not circular data it is supported by the app to plot non-circular data and
compute circular-linear correlations.

Panel: Data preparation
-----------------------

The first panel allows upload of single csv files.


Panel: Plot data
----------------

The computation of the polarity index is based on the following formula:

.. math::

    PI = \sqrt{ \left(\frac{1}{N} \sum_{i=1}^N \cos(\theta_i) \right)^2 + \left(\frac{1}{N} \sum_{i=1}^N \sin(\theta_i)\right)^2 }

.. note::
    This documentation is still under development and will be extended later!

Panel: Correlation analysis
---------------------------



Panel: Compare
--------------


