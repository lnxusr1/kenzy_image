import argparse
import logging
import os
import json
import cv2
from . import __app_name__, __version__
from .core import detector


def doParseArg(argName, argValue, cfg):
    if argValue is not None:
        cfg[argName] = argValue

    return


def doParseArgs(cfg, ARGS):
    if not isinstance(cfg, dict):
        cfg = {}

    if ARGS.no_faces:
        cfg["defectFaces"] = False
    
    if ARGS.no_objects:
        cfg["detectObjects"] = False
    
    if ARGS.no_motion:
        cfg["detectMotion"] = False

    if ARGS.no_markup:
        cfg["imageMarkup"] = False

    doParseArg("scaleFactor", ARGS.scale_factor, cfg)

    # Faces
    doParseArg("defaultFaceName", ARGS.face_detect_default_name, cfg)
    doParseArg("faceModel", ARGS.face_detect_model, cfg)
    doParseArg("faceFontColor", ARGS.face_detect_font_color, cfg)
    doParseArg("faceOutlineColor", ARGS.face_detect_outline_color, cfg)

    if ARGS.no_face_names:
        cfg["showFaceNames"] = False

    # Objects
    doParseArg("objDetectCfg", ARGS.object_detect_config, cfg)
    doParseArg("objDetectModel", ARGS.object_detect_model, cfg)
    doParseArg("objDetectLabels", ARGS.object_detect_labels, cfg)
    doParseArg("objFontColor", ARGS.object_detect_font_color, cfg)
    doParseArg("objOutlineColor", ARGS.object_detect_outline_color, cfg)

    if ARGS.no_object_names:
        cfg["showObjectNames"] = False

    # Motion
    doParseArg("motionThreshold", ARGS.motion_detect_threshold, cfg)
    doParseArg("motionMinArea", ARGS.motion_detect_min_area, cfg)
    doParseArg("motionOutlineColor", ARGS.motion_detect_outline_color, cfg)

    return cfg


parser = argparse.ArgumentParser(
    description=__app_name__ + " v" + __version__,
    formatter_class=argparse.RawTextHelpFormatter,
    epilog='''To start the services try:\npython3 -m image_analyzer\n\nMore information available at:\nhttp://kenzy.ai''')

parser.add_argument('-c', '--config', default=None, help="Configuration file")
parser.add_argument('-v', '--version', action="store_true", help="Print Version")

startup_group = parser.add_argument_group('Startup Options')

startup_group.add_argument('--no-markup', action="store_true", help="Hide outlines and names")
startup_group.add_argument('--scale-factor', default=None, help="Image scale factor (decimal).  Values < 1 improve performance.")

face_group = parser.add_argument_group('Face Detection')

face_group.add_argument('--no-faces', action="store_true", help="Disable face detection")
face_group.add_argument('--face-detect-default-name', default=None, help="Set the Unknown face name")
face_group.add_argument('--face-detect-model', default=None, help="Model to leverage (hog or cnn)")
face_group.add_argument('--face-detect-font-color', default=None, help="Face names font color as tuple e.g. (0, 0, 255)")
face_group.add_argument('--face-detect-outline-color', default=None, help="Faces outline color as tuple e.g. (0, 0, 255)")
face_group.add_argument('--no-face-names', action="store_true", help="Hides the face names even if identified.")
face_group.add_argument('--faces', action="append", nargs=2, metavar=("path", "name"), help="Face image and name e.g. --face image.jpg LNXUSR1")

object_group = parser.add_argument_group('Object Detection')

object_group.add_argument('--no-objects', action="store_true", help="Disable object detection")
object_group.add_argument('--object-detect-config', default=None, help="Object detection configuration")
object_group.add_argument('--object-detect-model', default=None, help="Object detection inference model file")
object_group.add_argument('--object-detect-labels', default=None, help="Object detection inference model label files")
object_group.add_argument('--object-detect-font-color', default=None, help="Object names font color as tuple e.g. (0, 0, 255)")
object_group.add_argument('--object-detect-outline-color', default=None, help="Object detection outline color as tuple e.g. (0, 0, 255)")
object_group.add_argument('--no-object-names', action="store_true", help="Hides the object names even if identified.")

motion_group = parser.add_argument_group('Motion Detection')

motion_group.add_argument('--no-motion', action="store_true", help="Disable motion detection")
motion_group.add_argument('--motion-detect-threshold', default=None, help="Motion detection difference threshold")
motion_group.add_argument('--motion-detect-min-area', default=None, help="Motion detection minimum pixel area")
motion_group.add_argument('--motion-detect-outline-color', default=None, help="Motion area outline color as tuple e.g. (0, 0, 255)")

logging_group = parser.add_argument_group('Logging Options')

logging_group.add_argument('--log-level', default="info", help="Options are full, debug, info, warning, error, and critical")
logging_group.add_argument('--log-file', default=None, help="Redirects all logging messages to the specified file")

ARGS = parser.parse_args()

if ARGS.version:
    print(__app_name__, "v" + __version__)
    quit()

logLevel = logging.INFO
if ARGS.log_level is not None and ARGS.log_level.strip().lower() in ["debug", "info", "warning", "error", "critical"]:
    logLevel = eval("logging." + ARGS.log_level.strip().upper())
elif ARGS.log_level is not None and ARGS.log_level.strip().lower() == "full":
    logLevel = logging.DEBUG

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    datefmt='%Y-%m-%d %H:%M:%S %z',
    filename=ARGS.log_file,
    format='%(asctime)s %(name)-12s - %(levelname)-9s - %(message)s',
    level=logLevel)

logger = logging.getLogger("STARTUP")

cfg = {}
if ARGS.config is not None and os.path.isfile(ARGS.config):
    with open(ARGS.config, "r", encoding="UTF-8") as fp:
        cfg = json.load(fp)

cfg = doParseArgs(cfg, ARGS)

obj = detector(**cfg)


if ARGS.faces is not None and isinstance(ARGS.faces, list):
    for item in ARGS.faces:
        if not os.path.isfile(item[0]):
            raise FileNotFoundError("Could not find file: " + str(item[0]))

        obj.addFace(item[0], item[1])

cam = cv2.VideoCapture(0)
while True:

    ret, frame = cam.read()
    obj.analyze(frame)
    print("=======================")
    print("FACES     =", len(obj.faces))
    print("OBJECTS   =", [x["name"] for x in obj.objects])
    print("MOVEMENTS =", True if len(obj.movements) > 0 else False)

    cv2.imshow('IMAGE_ANALYZER', obj.image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()