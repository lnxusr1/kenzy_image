# Changelog

All notable changes to this project will be documented in this file.

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