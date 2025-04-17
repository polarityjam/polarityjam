.. _methods:

Methods
=======
.. role:: raw-html(raw)
    :format: html

The methods that are used in Polarity-JaM are listed on this page. Whenever necessary, a brief summary
of the methodology is provided.

Segmentation
++++++++++++

Segmentation can currently be performed using the following methods:

- Cellpose `cellpose <https://github.com/MouseLand/cellpose>`_
- Segment Anything `segmentanything <https://segment-anything.com/>`_
- Mesmer deepcell `mesmer deepcell <https://github.com/vanvalenlab/deepcell-tf/tree/master>`_
- microSAM `microSAM <https://github.com/computational-cell-analytics/micro-sam>`_

Polarity-JaM natively uses Cellpose. Cellpose is a generalist algorithm for cell and nucleus segmentation.
Cellpose uses a neural network that was trained to predict horizontal and vertical gradients of
topological maps, together with a binary map indicating whether or not a pixel is inside a region
of interest. The topological maps were previously created with the help of ground-truth masks.
Following the combined gradients in a process known as gradient tracking, grouping together
pixel that converge to the same point and combining results with the information from the binary mask,
precise cell shapes can be recovered.

Alternatively, the user can try the following segmentation methods:
Segment Anything Model (SAM) is a new AI model from Meta AI that can "cut out" any object, in any image,
with a single click. For more information, please visit `segmentanything <https://segment-anything.com/>`_.

Mesmer deepcell is a deep learning model for cell segmentation (whole-cell and nuclear). For more information,
please visit `mesmer deepcell <https://github.com/vanvalenlab/deepcell-tf/tree/master>`_.

microSAM is a tool for segmentation and tracking in microscopy build on top of SegmentAnything. For more information,
please visit `microSAM <https://github.com/computational-cell-analytics/micro-sam>`_.

Cell properties
+++++++++++++++

Most cell properties are extracted from a `scikit-image <https://scikit-image.org/>`_ python library.
More specifically, we use the `regionprops <https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops>`_
module of scikit-image. It allows the user to measure properties of labeled image regions.


Region adjacency graph (neighborhood statistics)
++++++++++++++++++++++++++++++++++++++++++++++++

Our neighborhood statistic is calculated with the aid of a python module called `pysal <https://pysal.org/>`_  in
combination with the scikit-image graph implementation. Each cell is modelled as a node in the graph.
Additionally, a feature of interest (FOI) can be specified. This feature is included in the graph
structure and a `morans I <https://en.wikipedia.org/wiki/Moran%27s_I>`_ correlation analysis can be performed.



Other
+++++

Coding: `python <https://www.python.org/>`_ :raw-html:`<br />`
Documentation build: `sphinx <https://www.sphinx-doc.org/en/master/index.html>`_ :raw-html:`<br />`
Documentation hosting: `readthedocs <https://readthedocs.org/>`_ :raw-html:`<br />`
Jupyter notebooks: `jupyter <https://jupyter.org/>`_ :raw-html:`<br />`
Calculations: `numpy <https://numpy.org/>`_ :raw-html:`<br />`
Datasets: `pandas <https://pandas.pydata.org/>`_ :raw-html:`<br />`
Plot: `matplotlib <https://matplotlib.org/>`_ and `cmocean <https://pypi.org/project/cmocean/>`_ :raw-html:`<br />`
Logging: `python logging <https://docs.python.org/3/howto/logging.html>`_ :raw-html:`<br />`
Testing: `python unittest <https://docs.python.org/3/library/unittest.html>`_ :raw-html:`<br />`
Version control: `git <https://git-scm.com/>`_ :raw-html:`<br />`
Continuous integration: `github actions <https://github.com/features/actions>`_ :raw-html:`<br />`
Sketching: `inkscape <https://inkscape.org/>`_ :raw-html:`<br />`
Data Management: `omero <https://www.openmicroscopy.org/omero/>`_ :raw-html:`<br />`

