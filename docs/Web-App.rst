Web-App
========

We provide a simple means for statistical analysis of circular data that supports both exploratory analysis and the generation of graphs for publication. The App is based on the R Shiny inteface and can be started locally on you computer or on an web service provided by the Max-Delbrück-Center for MolecularMedicine.

Circular statistics
-------------------

The App provides a set of functions for the analysis and plotting of circular data. Generally circular data are data
that can be mapped on a circular scale, such as direction or time of day. Unlike linear data, which is measured on a
straight line scale, circular data has a natural periodicity, meaning that it repeats itself after a certain point.
Besides cellular polarity, examples outside the biology are the direction of the wind or
the direction of the magnetic field.

Circular data presents some unique challenges for statistical analysis because traditional statistical methods may not
be appropriate for this type of data. For instance, computing the average or mean of circular data by summing up the
values and dividing by the number of observations will often provide wrong results.


Here, we generally distinguish between directional, axial and linear data (or non-periodic data).
Directional data are data with values in [0, 2:math:`\pi` ] or 0 to 360, in radians or degrees, respectively.
Axial data are data with values [0, :math:`\pi` ] or 0 to 180 in degrees, meaning that it repeats itself after 180 degrees.
Linear data is not circular data it is supported by the app to plot non-circular data and
compute circular-linear correlations.

Panel: Data preparation
-----------------------

The first panel allows upload of single csv files. The file must contain a sample column (default: "label"),
which is numeric and contains whole numbers. Feature columns can have any name, but must contain numeric values,
meaning integers or floats.
any name. Additionally, column for grouping conditions must be specified, it must contain a limited amount
categorical values, so no numeric values.

The data can be can be filtered by removing samples of a certain condition in the field "Identifier of conditions".
Furthermore, numeric values can be filtered by setting and upper and lower threshold based on a column containing
numeric values.


Panel: Plot data
----------------

The user is asked to select a feature column and needs to select the type of data "directional", "axial" or "linear",
which determines the statistical analysis and the available plot options. We will discuss the most important computations.

For directional (circular) data mean and polarity index are computed from the resultant vector. We obtain the resultant
vector from N circular measurements by summing them component-wise.

.. math::

    \vec{R} = \left( \sum_{i=1}^N \cos(\alpha_i), \sum_{i=1}^N \sin(\alpha_i) \right) = (C, S),

where :math:`\alpha_i` i-th measurement given in the feature column. The computation of the polarity index defined by
the length of the resultant vector R:

.. math::

    PI = \| \vec{R} \| = \sqrt{ \left(\frac{1}{N} \sum_{i=1}^N \cos(\theta_i) \right)^2 + \left(\frac{1}{N} \sum_{i=1}^N \sin(\theta_i)\right)^2 }

The polarity index takes values between 0 and 1, where 0 indicates no directional polarity and
1 indicates perfect polarity.

The direction of the resultant vector,
which is proposed as the circular mean direction is denoted :math:`\bar{alpha}' and defined by

.. math::

    \bar{\alpha} = arctan*(S/C)

In the app computation arctan is computed as

.. math::

    \begin{cases}
        arctan*(S/C) = atan2 (S/C) & \text{ if S/C } \geq 0, \\
        arctan*(S/C) = atan2 (S/C) + 2\pi & \text{ if S/C < 0 }
    \end{cases}

in order to obtain 0 to 2:math:`\pi` (0 to 360). The atan2 is the common function defined in
https://en.wikipedia.org/wiki/Atan2 or https://search.r-project.org/CRAN/refmans/raster/html/atan2.html .






Panel: Correlation analysis
---------------------------



Panel: Compare
--------------

.. note::
    This documentation is still under development and will be extended later!
