# Changelog

All notable changes to this project will be documented in this file.

## [0.0.6]

### Added

- ```detector.analyze()``` now supports overrides to core setting by supplying the boolean values for detectFaces, detectObjects, and/or detectMotion directly to this function.
- ```objDetectList``` now enables you to specify a list of objects to look for (must be one of the values in the object labels list).
- Enabled ```--camera-device``` for cli calls to specify what device or RTSP stream to use as the source.
- ```objModelType``` now enables `yolo` or `ssd` options for model usage.  (Yolo leverages yolov7-tiny)

### Modified

- Default object detection model changed from MobileNet V3 SSD to Yolov7
- Fixed spelling mistakes in documentation for option variables
- Optimized hierarchy for image processing to improve overall performance and lag

## [0.0.5]

### Added

- ```orientation``` option with values of 0, 90, 180, or 270 to rotate the input image by the desired degree.

### Modified

- Changed dependency from opencv-python to opencv-contrib-python (Raspberry Pi requires this for Face/Object detection)
- Cleaned up mistake in Detector documentation with invalid object name in example

## [0.0.4]

### Modified

- Fixed but in rt_secs/rtSets for runtime stats.


## [0.0.3]

### Modified

- Multiple updates to documentation
- Corrected a bug when setting font and outline colors using tuples for running as module

## [0.0.2]

This is the initial version and has not be heavily validated or tested.  Stable versions should be available in v1.0 or later.

### Added

- CLI Parameters to enable calling module directly (python -m kenzy_image)
- Default to enable all detections (Face, Object, and Motion)
- Arguments added to enable faces to be recognized e.g. ```--faces /path/image1.jpg LNXUSR1```  (Can add multiple ```--faces``` arguments.)
- Added ```--config``` to enable specifying a JSON formatted file with all options configured.  (See docs for more info.)
- Use ```--help``` for full option list

### Modified

- Renamed module to kenzy_image