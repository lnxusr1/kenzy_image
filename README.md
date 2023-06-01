# kenzy_image &middot; [![GitHub license](https://img.shields.io/github/license/lnxusr1/image_analyzer)](https://github.com/lnxusr1/kenzy_image/blob/master/LICENSE) ![Python Versions](https://img.shields.io/pypi/pyversions/yt2mp3.svg) ![Read the Docs](https://img.shields.io/readthedocs/kenzy_image) ![GitHub release (latest by date)](https://img.shields.io/github/v/release/lnxusr1/kenzy_image)

This module is dedicated to simplifying the interactions required for face detection, face recognition, object detection, and motion detection.  Visit our main site: [https://kenzy.ai/](https://kenzy.ai/)

More info available in the [documentation](https://image-docs.kenzy.ai/)

## Installation

The easiest way to install kenzy_image is with the following:

```
pip install kenzy-image
```

Just make sure you're running Python 3.6 or newer.

## Embedding into your program

Visit the [detector](https://image-docs.kenzy.ai/en/latest/detector/) page in the [documentation](https://image-docs.kenzy.ai/) for more information.

```
import cv2  # For webcam input
from kenzy_image import detector

# Create our Kenzy Detector Object
myImageDetector = detector(detectFaces=True, 
                           detectObjects=True, 
                           objModelType="ssd",  # or yolo
                           detectMotion=True, 
                           imageMarkup=True, 
                           scaleFactor=0.5)

# Add Named Faces to Recognize
myImageDetector.addFace("IMG_4291_portrait_jon.jpg", "Jon Doe")
myImageDetector.addFace("IMG_7033_portrait_jane.jpg", "Jane Doe")

# Open the camera stream
cam = cv2.VideoCapture(0)

while True:

    # Read frame from camera
    ret, frame = cam.read()

    # Analyze with kenzy_image
    myImageDetector.analyze(frame)

    print("=======================")
    print("FACES     =", len(myImageDetector.faces))
    print("OBJECTS   =", [x.get("name") for x in myImageDetector.objects])
    print("MOVEMENTS =", True if len(myImageDetector.movements) > 0 else False)

    # Show the frame with markup
    cv2.imshow('KENZY_IMAGE', myImageDetector.image)

    # Loop until [Esc] or "q"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
```

## Running as module

Options are as follows for starting kenzy_image:

```
python -m kenzy_image [OPTIONS]
```

For example, to switch the model to MobileNet v3's SSD model use:
```
python -m kenzy_image --object-detect-type ssd  
```

Use ```--help``` for more information on options.

## References and Other Useful Links

Many thanks to those that build the models and core libraries that KENZY_IMAGE incorporates.  Please find links to those below:

#### Face Detection & Recognition

- [Face Recognition Concepts and Examples](https://git.ece.iastate.edu/se_329_cylicon_valley/face_recognition)
- [Face Recognition Library](https://github.com/ageitgey/face_recognition)

#### Object Detection

- [COCO Inference Model &amp; Config](https://github.com/zafarRehan/object_detection_COCO)
- [COCO Labels](https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/) <-- Better label set for frozen inference model
- [YOLOv7 Model](https://github.com/wongkinyiu/yolov7) <-- Source for the included [Yolov7-tiny.pt](https://github.com/WongKinYiu/yolov7/releases)
- [YOLOv7 Python Library](https://github.com/akashAD98/yolov7-pip-1)

### Motion Detection

- [Motion Detection Example](https://towardsdatascience.com/image-analysis-for-beginners-creating-a-motion-detector-with-opencv-4ca6faba4b42)

-----

## Help &amp; Support
Help and additional details is available at [https://kenzy.ai](https://kenzy.ai)

Read the docs: [https://image-docs.kenzy.ai/](https://image-docs.kenzy.ai/)