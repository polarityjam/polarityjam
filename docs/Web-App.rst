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
Directional data are data with values in [0, :math:`2\pi` ] or 0 to 360, in radians or degrees, respectively.
Axial data are data with values [0, :math:`\pi` ] or 0 to 180 in degrees, meaning that it repeats itself after 180
degrees. Linear data is not circular data it is supported by the app to plot non-circular data and
compute circular-linear correlations.

Panel: Data preparation
-----------------------

The first panel allows upload of single csv files. The file must contain a sample column (default: "label"),
which is numeric and contains whole numbers. Feature columns can have any name, but must contain numeric values,
meaning integers or floats.
any name. Additionally, a column for grouping conditions must be specified, it must contain a limited amount
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

    PI = \| \vec{R} \| = \sqrt{ \left(\frac{1}{N} \sum_{i=1}^N \cos(\alphai) \right)^2
                                                            + \left(\frac{1}{N} \sum_{i=1}^N \sin(\alpha_i)\right)^2 }

The polarity index takes values between 0 and 1, where 0 indicates no directional polarity and 1 indicates perfect polarity.

The direction of the resultant vector, which is proposed as the circular mean direction is denoted :math:`\bar{\alpha}`
and defined by

.. math::

    \bar{\alpha} = arctan*(S/C)

In the app computation arctan is computed as

.. math::

    arctan*(S/C) = \begin{cases}
         atan2 (S/C) & \text{ if S/C } \geq 0, \\
        atan2 (S/C) + 2\pi & \text{ if S/C < 0 }
    \end{cases}

in order to obtain 0 to :math:`2\pi` (0 to 360). The atan2 is the common function defined in
https://en.wikipedia.org/wiki/Atan2 or https://search.r-project.org/CRAN/refmans/raster/html/atan2.html .

Assume that there is an expected direction, for instance, in case of nuclei-Golgi polarity we might expect polarization
against blood flow, we can derive a measure of the deviation of the observed direction from the expected direction, which is called
the V-score. The V-score is defined as

.. math::

    V = \| \vec{R} \| cos (\bar{\alpha} - \bar{\mu}),

where :math:`\bar{\mu}` is the expected direction. The V-score takes values between -1 and 1, where -1 indicates polarization
against the expected direction and 1 indicates perfect polarization along the expected direction.
0 indicates no polarization perpendicular to the expected direction. We therefore obtain a signed polarity index (PI).

Angular data describes the orientation of an "axis", as for instance, the long axis of cells or nuclei, rather than a direction.
These observations of axes orientation are referred to as axial data. The axial data are handled by "doubling the
angles", meaning transforming each angle :math:`\alpha_i` to :math:`2\alpha_i` which removes the directional ambiguity.
Assuming a vector from N axial measurements  :math:`\alpha_i, i=1, \dots, N`,  we first obtain the mean :math:`\bar{\alpha}*`
and polarity index from the doubled angles :math:`2\alpha_i, i=1, \dots, N` as described above. The axial mean
computed from :math:`\bar{\alpha} =  \frac{1}{2} \bar{\alpha}*` and the axial polarity index and V-score is directly
computed from the doubled values.

For both directional and axial data the variance is computed from

.. math::
    S = 1 - PI

Multiple quantities have been introduced as analogues to the linear standard deviation.
We compute the angular deviation, which is given by

.. math::

    s_a = \sqrt{1 - PI}

with values in the interval [0, :math:`\sqrt{2}`], and the circular standard deviation, which is defined as
defined as

.. math::

    s_c = \sqrt{ - 2 ln PI}

and ranges from 0 to :math:`\infty` :cite:t:`berens2009circstat`.

All statistical data including the mean, standard deviation, polarity index, angular standard deviation and circular
standard, percentile are computed for each condition and can be downloaded.

Note, that for linear data the mean and standard deviation are computed from the usual sample mean and standard
deviation, which is not further discussed here.



Panel: Correlation analysis
---------------------------



Panel: Compare
--------------

For further reading we recommend:

.. bibliography::

.. note::
    This documentation is still under development and will be extended later!
