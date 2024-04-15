WormTracer

WormTracer is an algorithm designed to accurately determine the centerline of a worm in time-lapse images. Unlike conventional methods that analyze individual images separately, WormTracer estimates worm centerlines within a sequence of images concurrently. This process enables the resolution of complex postures that are difficult to assess when treated as isolated images. The centerlines obtained through WormTracer exhibit higher accuracy compared to those acquired using conventional methods.

WormTracer codes are written in Python and provided in two forms, as an IPython Notebook (.ipynb), or a Python source code (.py). Users can choose either format depending on their own environment for running the codes. Please refer to (3) or (4) respectively for each environment.

- Reference -
"WormTracer: A precise method for worm posture analysis using temporal continuity"
bioRxiv 2023.12.11.571048; doi: https://doi.org/10.1101/2023.12.11.571048

and a manuscript under review.

<<< How to use WormTracer >>>

In all cases, input images need to be time-lapse images in a folder, either as a single multipage tiff file or serial numbered image files with a format of your choice, such as tif, png, jpeg etc.
In all cases, results are saved in the input image folder.

(1) Data required

WormTracer only requires a "folder containing binarized worm video images".
Put a binarized worm video in a single folder, either as a single multipage tiff file or serial numbered image files with a format of your choice, such as tif, png, jpeg etc that can be read by OpenCV. Binarization threshold is sometimes critical (see Tips), so we recommend users binarize raw images manually, for example using the "Image > Adjust > Threshold" function of ImageJ. The folder name is arbitrary and you will specify the folder name when you run WormTracer. For the serial numbered format, image file names should include numbers at the end in a chronological order; you can save the images in this format for example by selecting “File > Save as > Image Sequence...” in ImageJ. Alternatively, stack images can be saved as a multipage tiff file by "File > Save as > Tiff..." in ImageJ. Put this file alone in a new folder.

Tips: For a movie that include a high rate of bent postures, for example of a loopy mutant or a worm that shows high rate of omega turns, relatively strict threshold (smaller worm area and thin body) is recommended, even though holes are seen in the binarized worm body. On the other hand, in cases where accurate positioning of the centerline at the head and tail tips are required, a threshold at which these shapes appear clearly is recommended.

(2) Results

In all cases, results are saved in the input image folder.

(3) Running WormTracer from IPython Notebook, such as on Jupyter Notebook or Google Colaboratory (GUI environment)

(preparation)
Make input images in a folder
Place functions.py at an appropriate path
Place wt18_5.ipynb and change parameters, especially the path to functions.py and path to the input image folder.

(execution)
Run all cells in wt18_5.ipynb

(4) Running WormTracer from command line (CLI)

(installation)
Install WormTracer by entering the following command after activating appropriate python environment
$ pip install git+https://github.com/yuichiiino1/WormTracer.git#subdirectory=WormTracer18_5

Alternatively, the codes can be downloaded from https://github.com/yuichiiino1/WormTracer/ (current version is in WormTracer18_5 subfolder), either by pip clone or https download, and (after optional modification) install as
$ pip install [local path to WormTracer18_5 folder]

Confirm successfule installation by making sure that WormTracer appears by $ pip list
In case you need to uninstall, just enter $ pip uninstall WormTracer

(preparation)
Make input images in a folder
Make a parameter file xxxx.yaml according to your preferred options, or use the sample file, config.yaml, as is.

(execution in case of python interactive mode)
Enter following commands:
(python)$ from WormTracer import wt
(python)$ wt.run('[path to xxxx.yaml]', '[path to input image folder]', parameter1=1000)
Note: after two mandatory arguments, optional (parameter_name, value) pairs (example: parameter1=1000) can be specified, to override the value of a specific parameter(s).

(execution in case of batch mode)

< wt0000.sh > (adjust according to your environment)
#!/bin/sh
python3 wtexe.py [path to xxxx.yaml] [path to input image folder]

< wtexe.py >
import sys
from WormTracer import wt
wt.run(sys.argv[1], sys.argv[2])

Optional parameter pairs can be specfied also in this case.

(Troubleshooting)
Under some environment, saving the results as a mp4 movie may fail, with an error message such as "unknown file extension: .mp4". In this case, try downloading the ffmpeg package from https://ffmpeg.org/download.html, and specify the path to ffmpeg executable in wtexe.py as follows.

from matplotlib import pyplot as plt
plt.rcParams["animation.ffmpeg_path"] = "PATH_TO_FFMPEG"

before "wt.run(sys.argv[1], sys.argv[2])"

Or, just avoid making the movie file by setting the parameter SaveCenterlinedWormsMovie as False in xxxx.yaml.



