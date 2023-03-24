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

Circular data presents some unique challenges for statistical analysis because traditional statistical methods may not
be appropriate for this type of data. For instance, computing the average or mean of circular data by summing up the
values and dividing by the number of observations will often provide wrong results.


Here, we generally distinguish between directional, axial and linear data (or non-periodic data).
Directional data are data with values in [0,2 &pi ] or 0 to 360, in radians or degrees, respectively.
Axial data are data with values [0, &pi] or 0 to 180 in degrees, meaning that it repeats itself after 180 degrees.
Linear data is not circular data it is supported by the app to plot non-circular data and
compute circular-linear correlations.

Panel: Data preparation
-----------------------

The first panel allows upload of single csv files. The file must contain a sample column (default: "label"),
which is numeric and contains whole numbers. Furthermore a column with numeric float values, the column can have
any name. Also a column for grouping of conditions must be specified, it must contain a limited amount
categorial values, so no numeric values.

The data can be can be filtered by removing samples of a certain condition in the field "Identifier of conditions".
Furthermore, numeric values can be filtered by setting and upper and lower threshold based on a column containing
numeric values.


Panel: Plot data
----------------

The computation of the polarity index is based on the following formula:

.. math::

    PI = \sqrt{ \left(\frac{1}{N} \sum_{i=1}^N \cos(\theta_i) \right)^2 + \left(\frac{1}{N} \sum_{i=1}^N \sin(\theta_i)\right)^2 }



Panel: Correlation analysis
---------------------------



Panel: Compare
--------------

.. note::
    This documentation is still under development and will be extended later!
