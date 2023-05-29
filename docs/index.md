# kenzy_image &middot; [![GitHub license](https://img.shields.io/github/license/lnxusr1/image_analyzer)](https://github.com/lnxusr1/kenzy_image/blob/master/LICENSE) ![Python Versions](https://img.shields.io/pypi/pyversions/yt2mp3.svg) ![Read the Docs](https://img.shields.io/readthedocs/kenzy_image) ![GitHub release (latest by date)](https://img.shields.io/github/v/release/lnxusr1/kenzy_image)

This module is dedicated to simplifying the interactions required for face detection, face recognition, object detection, and motion detection.

## Installation

The easiest way to install kenzy_image is with the following:

```
pip install kenzy-image
```

Just make sure you're running Python 3.6 or newer.

## Embedding into your program

Visit the [detector](detector.md) page

## Running as module

Options are as follows for starting kenzy_image:

```
python -m kenzy_image [OPTIONS]

General Options:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        Configuration file
  -v, --version         Print Version

Startup Options:
  --no-markup           Hide outlines and names
  --scale-factor SCALE_FACTOR
                        Image scale factor (decimal).  Values < 1 improve performance.

Face Detection:
  --no-faces            Disable face detection
  --face-detect-default-name FACE_DETECT_DEFAULT_NAME
                        Set the Unknown face name
  --face-detect-model FACE_DETECT_MODEL
                        Model to leverage (hog or cnn)
  --face-detect-font-color FACE_DETECT_FONT_COLOR
                        Face names font color as tuple e.g. (0, 0, 255)
  --face-detect-outline-color FACE_DETECT_OUTLINE_COLOR
                        Faces outline color as tuple e.g. (0, 0, 255)
  --no-face-names       Hides the face names even if identified.
  --faces path name     Face image and name e.g. --face image.jpg LNXUSR1

Object Detection:
  --no-objects          Disable object detection
  --object-detect-config OBJECT_DETECT_CONFIG
                        Object detection configuration
  --object-detect-model OBJECT_DETECT_MODEL
                        Object detection inference model file
  --object-detect-labels OBJECT_DETECT_LABELS
                        Object detection inference model label files
  --object-detect-font-color OBJECT_DETECT_FONT_COLOR
                        Object names font color as tuple e.g. (0, 0, 255)
  --object-detect-outline-color OBJECT_DETECT_OUTLINE_COLOR
                        Object detection outline color as tuple e.g. (0, 0, 255)
  --no-object-names     Hides the object names even if identified.

Motion Detection:
  --no-motion           Disable motion detection
  --motion-detect-threshold MOTION_DETECT_THRESHOLD
                        Motion detection difference threshold
  --motion-detect-min-area MOTION_DETECT_MIN_AREA
                        Motion detection minimum pixel area
  --motion-detect-outline-color MOTION_DETECT_OUTLINE_COLOR
                        Motion area outline color as tuple e.g. (0, 0, 255)

Logging Options:
  --log-level LOG_LEVEL
                        Options are full, debug, info, warning, error, and critical
  --log-file LOG_FILE   Redirects all logging messages to the specified file

To start the services try:
python3 -m kenzy_image

More information available at:
http://kenzy.ai
```

-----

## Help &amp; Support
Help and additional details is available at [https://kenzy.ai](https://kenzy.ai)
