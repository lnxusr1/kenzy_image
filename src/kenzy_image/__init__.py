"""
More info at Kenzy.Ai
"""
try:
    import sys
    import os
    import logging 
    import traceback
    from core import detector
except ModuleNotFoundError:
    logging.debug(str(sys.exc_info()[0]))
    logging.debug(str(traceback.format_exc()))
    logging.info("Unable to start analyzer due to missing libraries")

__app_name__ = "kenzy_image"
__app_title__ = "kenzy_image"

with open(os.path.join(os.path.dirname(__file__), "VERSION"), "r", encoding="UTF-8") as fp:
    __version__ = fp.readline().strip()

VERSION = [(int(x) if x.isnumeric() else x) for x in __version__.split(".")]
