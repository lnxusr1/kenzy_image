import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__).rstrip("/\\")[0:-5] + "src"))

def test_detector_init_basic_on():
    try:
        import kenzy_image
        import kenzy_image.core
        d = kenzy_image.core.detector(imageMarkup=True, detectFaces=True, detectMotion=True, detectObjects=True)

        assert d._imageMarkup and d._detectFaces and d._detectMotion and d._detectObjects
    except Exception:
        assert False

def test_detector_init_basic_off():
    try:
        import kenzy_image
        import kenzy_image.core
        d = kenzy_image.core.detector(imageMarkup=False, detectFaces=False, detectMotion=False, detectObjects=False)

        assert not d._imageMarkup and not d._detectFaces and not d._detectMotion and not d._detectObjects
    except Exception:
        assert False
