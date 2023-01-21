import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__).rstrip("/\\")[0:-5] + "src"))


def test_detector_fileNotFound_addFace():
    try:
        import kenzy_image
        import kenzy_image.core
        d = kenzy_image.core.detector()
        d.addFace("/tmp/FILE_NOT_FOUND.jpg", "LNXUSR1")
    
        assert False

    except FileNotFoundError:
        assert True 

    except Exception:
        assert False

def test_detector_fileNotFound_formatImage():
    try:
        import kenzy_image
        import kenzy_image.core
        d = kenzy_image.core.detector()
        d._formatImage("/tmp/FILE_NOT_FOUND.jpg")
    
        assert False

    except FileNotFoundError:
        assert True 

    except Exception:
        assert False