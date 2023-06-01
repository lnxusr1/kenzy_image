# kenzy_image.detector

## Basic Usage

```
import cv2  # For webcam input
from kenzy_image.core import detector

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

## detector Constructor

| Variable | Default | Description |
| :------- | :------ | :---------- |
| orientation | `0` | Image orientation. (0, 90, 180, or 270) |
| detectFaces | `True` | Enable/disable face detection |
| detectObjects | `True` | Enable/disable object detection |
| detectMotion | `True` | Enable/disable motion detection |
| imageMarkup | `True` | Enables drawing outlines on __detector.image__ |
| scaleFactor | `1.0` | Scale coefficient to apply to original image <br /><small>0.75 = scale source image to 75% to improve performance</small> |
| defaultFaceName | `"Unknown"` | The label to use for any faces detected but not recognized |
| faceModel | `"hog"` | Face model to use (__hog__ or __cnn__)
| faceFontColor | `(255, 255, 255)` | Color as RGB tuple for text of face name |
| faceOutlineColor | `(0, 0, 255)` | Color as RGB tuple for box around the face |
| showFaces | `True` | Enable drawing outlines around faces on __detector.image__ |
| objModelType | `ssd` | Sets the model type (can be __yolo__ or __ssd__) |
| objDetectCfg | String | Sets the path to the configuration for the inference model |
| objDetectModel | String | Sets the inference model to use for objects |
| objDetectLabels | String | Sets the path to the object labels list file |
| objDetectList | String | Limits list of objects to detect detection (optional) |
| objFontColor | `(255, 255, 255)` | Color as RGB tuple for text of object name |
| objOutlineColor | `(255, 0, 0)` | Color as RGB tuple for box around the object |
| showObjectNames | `True` | Enable drawing outlines around objects on __detector.image__ |
| motionThreshold | `20` | Threshold of color change between frames indicating motion |
| motionMinArea | `50` | Minimum pixel area of change indicating motion |
| motionOutlineColor | `(255, 0, 0)` | Color as RGB tuple for box around the motion |

## Return Values

The `detector.analyze()` processes the supplied image and saves the results into variables as defined below:

| Variable | Type | Description |
| :------- | :------ | :---------- |
| detector.faces | list | Listing of faces detected, their location in the image, <br />and their name if known |
| detector.objects | list | Listing of objects detected, their location within the image, <br /> and their type (e.g. "cellphone") |
| detector.movements | list | Listing of all areas where movement was identified <br /> with their bounding boxes in the image |

In the Basic Usage section above it shows a very simple example of how these values could be leveraged.

## Yolov7 - objDetectCfg

For `yolo` models you can supply your own configuration to change the confidence score, IOU, and channels.

```
import cv2  # For webcam input
from kenzy_image.core import detector

# Create our Kenzy Detector Object
myImageDetector = detector(detectFaces=True, 
                           detectObjects=True, 
                           objModelType="yolo",
                           objDetectModel="/path/to/yolov7-tiny.pt",
                           objDetectLabels="/path/to/yolov7-labels.txt",
                           objDetectCfg={
                               "libary": "yolov7",
                               "confidence": 0.5,
                               "iou": 0.45,
                               "size": 640,
                               "augment": False
                           },
                           detectMotion=True, 
                           imageMarkup=True, 
                           scaleFactor=0.5)
```

## Optimizing Performance & Results

* Object detection is currently using __Yolov7__ with the COCO data set from the *yolov7-tiny* model.
* To use motion detection you must have at least 2 image frames sent via `.analyze()` as it calculates the difference between the two images to identify motion.  For best results consider using a video or camera feed.
* You may need to increase the `motionMinArea` to reduce chatter
* Shrinking the __scaleFactor__ to a decimal between `0` and `1` will improve performance at the cost of accuracy by scaling down the image.  Likewise, if you increase this value beyond `1` your results may improve *marginally* at the expense of performance.