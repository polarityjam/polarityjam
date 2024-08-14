.. _faq:

Frequently Asked Questions
===========================
.. role:: raw-html(raw)
    :format: html

How to bring your own segmentation?
-----------------------------------
To bring your own segmentation you need to provide a `_seg.npy` file that
lives in the same folder as the images you want to process.

So for example when your image is named `image.tif`, the segmentation file should be named `image_seg.npy`.

The segmentation file should be a numpy array with the same shape as the
image and should contain the segmentation labels with `0` indicating the background.

Please make sure your npy file has an item with the key `masks` that contains the segmentation labels.

How does such a segmentation file look?
---------------------------------------
You can look at an example `_seg.npy` file by first downloading our example data `here <https://github.com/polarityjam/polarityjam/blob/main/src/polarityjam/test/resources/data.zip>`_.
Extracting the zip file and looking at the `_seg.npy` files in the `data/golgi_nuclei/set1/` folder.

Here is a short code block to load the segmentation file and visualize it:
```python
import numpy as np
import matplotlib.pyplot as plt

seg = np.load("path/to/your/seg.npy")
plt.imshow(seg.item()["masks"])
plt.show()

```

How to bring your own model?
----------------------------
Currently, you can only use your own cellpose model.
To do so, you need to provide the path to the model in the parameter file `model_path`.

You can train your own model for example using the cellpose `GUI <https://cellpose.readthedocs.io/en/latest/gui.html>`_ .

What is a parameter file?
-------------------------
A parameter file is a JSON file that defines the parameters for:

- feature extraction
- visualization
- input

Additionally, it can be used to define the parameters used with the segmentation algorithm.

The definitions in my parameter file are ignored
------------------------------------------------
When you provide a parameter file, the parameters in the file will overwrite the default parameters.
Moreover, if you load a parameter file using the python API after you changed specific parameters,
the parameters in the file will overwrite the parameters you already changed.

Make sure you load first before you change the parameters manually.


How does a parameter file look?
-------------------------------
You can find our default parameter file `here: <https://github.com/polarityjam/polarityjam/blob/main/src/polarityjam/utils/resources/parameters.yml>`_.
You find the segmentation parameter file for the algorithms we support here:

- `cellpose <https://github.com/polarityjam/polarityjam/blob/main/src/polarityjam/segmentation/cellpose.yml>`_
- `deepcell <https://github.com/polarityjam/polarityjam/blob/main/src/polarityjam/segmentation/deepcell.yml>`_
- `microsam <https://github.com/polarityjam/polarityjam/blob/main/src/polarityjam/segmentation/microsam.yml>`_
- `sam <https://github.com/polarityjam/polarityjam/blob/main/src/polarityjam/segmentation/sam.yml>`_

They also list all the parameters you can play around with.
Please refer to the original documentation of the segmentation algorithm for more information on the parameters.


What is a key file?
-------------------
A key file is a CSV file that defines the data structure of the input.
It enables data migration without modifying the CSV itself, as the file paths
are relative to a specified root folder (e.g., inputpath) that you provide
along with the key file during input.

The key file can be used with the `polarityjam` CLI option `run-key`.

How does a key file look?
-------------------------

A key file is a CSV file with the following columns:

+--------------+-------------+
| folder_name  | short_name  |
+==============+=============+
| set_1        | cond_1      |
+--------------+-------------+
| set_2        | cond_2      |
+--------------+-------------+

Specify a short_name different to the folder_name to rename each folder in the output.


Why are my paths in the key file not recognized?
------------------------------------------------
One reason could be that you are using the wrong path separator. On Windows, the path separator is a backslash, while on Unix systems it is a forward slash.
If you are using a Windows system, you need to escape the backslashes in the path.
For example, if you have a path like `C:\Users\user\Documents\keyfile.txt`, try to write it as `C:\\Users\\user\\Documents\\keyfile.txt`.
Also, when you swith between Windows and Unix systems, make sure to adjust the path separators accordingly. Linux uses forward slashes `/`.


