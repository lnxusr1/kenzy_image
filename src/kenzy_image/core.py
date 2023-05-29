import os
import face_recognition
import cv2
import numpy as np
import time
import logging


class detector(object):
    def __init__(self, **kwargs):

        self.logger = logging.getLogger("DETECTOR")
        self.rt_logger = logging.getLogger("DETECT_TIME")

        self._imageMarkup = kwargs.get("imageMarkup", True)
        self._faceNames = []
        self._faceEncodings = []

        orientation = int(kwargs.get("orientation", "0"))  # 0, 90, 180, 270
        self._orientation = None
        if orientation == 90:
            self._orientation = cv2.ROTATE_90_CLOCKWISE
            self._isRotated = False
        elif orientation == 180:
            self._orientation = cv2.ROTATE_180
            self._isRotated = False
        elif orientation == 270 or orientation == -90:
            self._orientation = cv2.ROTATE_90_COUNTERCLOCKWISE
            self._isRotated = False
        
        self._detectFaces = kwargs.get("detectFaces", True)
        self._recognizeFaces = False
        self._detectObjects = kwargs.get("detectObjects", True)
        self._detectMotion = kwargs.get("detectMotion", True)
        
        self.logger.info("Face Detection     = " + str("Enabled" if self._detectFaces else "Disabled"))
        self.logger.info("Object Detection   = " + str("Enabled" if self._detectObjects else "Disabled"))
        self.logger.info("Motion Detection   = " + str("Enabled" if self._detectMotion else "Disabled"))
        
        self._faceScaleDownFactor = float(kwargs.get("scaleFactor", "1.0"))  # (1.0 >= VALUE > 0)
        self._faceModel = kwargs.get("faceModel", "hog")  # hog or cnn
        self._faceScaleUpFactor = 1.0  
        self._defaultFaceName = kwargs.get("defaultFaceName", "Unknown")
        self._faceShowNames = kwargs.get("showFaceNames", True)
        self._faceOutlineColor = kwargs.get("faceOutlineColor", (0, 0, 255))
        self._faceFontColor = kwargs.get("faceFontColor", (255, 255, 255))
        
        if self._faceScaleDownFactor < 1 and self._faceScaleDownFactor > 0:
            self._faceScaleUpFactor = ((1.0 - float(self._faceScaleDownFactor)) / float(self._faceScaleDownFactor)) + 1.0

        self.image = None
        self._lastImage = None
        self._scaledBGRImage = None
        self._scaledRGBImage = None
        self._scaledBWImage = None
        self._lastScaledBWImage = None

        self._objConfigFile = kwargs.get("objDetectCfg", os.path.join(os.path.dirname(__file__), "resources", "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"))
        self._objModelFile = kwargs.get("objDetectModel", os.path.join(os.path.dirname(__file__), "resources", "frozen_inference_graph.pb"))
        self._objLabelFile = kwargs.get("objDetectLabels", os.path.join(os.path.dirname(__file__), "resources", "labels.txt"))
        self._objShowNames = kwargs.get("showObjectNames", True)
        self._objOutlineColor = kwargs.get("objOutlineColor", (255, 0, 0))
        self._objFontColor = kwargs.get("objFontColor", (255, 255, 255))

        self._objModel = cv2.dnn_DetectionModel(self._objModelFile, self._objConfigFile)
        self._objModel.setInputSize(320, 320)  # greater this value the better the results; tune it for best output
        self._objModel.setInputScale(1.0 / 127.5)
        self._objModel.setInputMean((127.5, 127.5, 127.5))
        self._objModel.setInputSwapRB(True)
        self._objLabels = []
        
        self._motionThreshold = kwargs.get("motionThreshold", 20)
        self._motionMinArea = kwargs.get("motionMinArea", 50)
        self._motionOutlineColor = kwargs.get("motionOutlineColor", (0, 255, 0))

        self.logger.debug("==< CONFIG >===================================")
        self.logger.debug("orientation        = " + str(orientation))
        self.logger.debug("detectFaces        = " + str(self._detectFaces))
        self.logger.debug("detectObjects      = " + str(self._detectObjects))
        self.logger.debug("detectMotion       = " + str(self._detectMotion))
        self.logger.debug("scaleFactor        = " + str(self._faceScaleDownFactor))
        self.logger.debug("faceModel          = " + str(self._faceModel))
        self.logger.debug("defaultFaceName    = " + str(self._defaultFaceName))
        self.logger.debug("showFaceNames      = " + str(self._faceShowNames))
        self.logger.debug("faceFontColor      = " + str(self._faceFontColor))
        self.logger.debug("faceOutlineColor   = " + str(self._faceOutlineColor))
        self.logger.debug("objDetectCfg       = " + str(self._objConfigFile))
        self.logger.debug("objDetectModel     = " + str(self._objModelFile))
        self.logger.debug("objDetectLabels    = " + str(self._objLabelFile))
        self.logger.debug("showObjectNames    = " + str(self._objShowNames))
        self.logger.debug("objFontColor       = " + str(self._objFontColor))
        self.logger.debug("objOutlineColor    = " + str(self._objOutlineColor))
        self.logger.debug("motionThreshold    = " + str(self._motionThreshold))
        self.logger.debug("motionMinArea      = " + str(self._motionMinArea))
        self.logger.debug("motionOutlineColor = " + str(self._motionOutlineColor))
        self.logger.debug("==</ CONFIG >==================================")

        self.faces = []
        self.objects = []
        self.movements = []

        self._loadLabels()
        
    def addFace(self, fileName, faceName):
        if not os.path.isfile(fileName):
            raise FileNotFoundError("Face file not found.")

        newFaceImage = face_recognition.load_image_file(fileName)
        newFaceEncoding = face_recognition.face_encodings(newFaceImage)[0]

        self._faceEncodings.append(newFaceEncoding)
        self._faceNames.append(str(faceName) if faceName is not None else "")

        self.logger.info("Adding face (" + str(faceName) + "): " + str(fileName))
        
        self.logger.debug("Enabling face detection because face image has been added")
        self._detectFaces = True
        self._recognizeFaces = True

        return True
        
    def _formatImage(self, image=None):
        if image is None:
            return

        self._lastScaledBWImage = self._scaledBWImage
        self.image = image
        
        if isinstance(image, str):
            if os.path.isfile(image):
                self.image = cv2.imread(image)
            else:
                raise FileNotFoundError("Image not found for analysis.")

        if self._orientation is not None:
            self.image = cv2.rotate(self.image, self._orientation)

        self._scaledBGRImage = None
        self._scaleImage()

        self._scaledBWImage = None
        self._bwImage()

    def _scaleImage(self):
        if self._scaledBGRImage is None:
            if self._faceScaleDownFactor != 1.0:
                self._scaledBGRImage = cv2.resize(self.image, (0, 0), fx=self._faceScaleDownFactor, fy=self._faceScaleDownFactor, interpolation=cv2.INTER_AREA)
            else:
                self._scaledBGRImage = self.image.copy()

            self._scaledRGBImage = cv2.cvtColor(self._scaledBGRImage, cv2.COLOR_BGR2RGB)
    
    def _bwImage(self):
        self._scaleImage()
        if self._scaledBWImage is None:
            self._scaledBWImage = cv2.cvtColor(self._scaledBGRImage, cv2.COLOR_BGR2GRAY)
            self._scaledBWImage = cv2.GaussianBlur(src=self._scaledBWImage, ksize=(5, 5), sigmaX=0)

    def _loadLabels(self):
        self.logger.debug("Loading label file from " + str(self._objLabelFile))
        with open(self._objLabelFile, 'rt') as fp:
            self._objLabels = fp.read().rstrip('\n').split('\n')

    def analyze(self, image, detectFaces=None, detectObjects=None, detectMotion=None):
        """
        Set detectFaces, detectObjects, or detectMotion to True or False to override global setting for this one image.
        """
        
        start = time.time()

        self.faces = []
        self.objects = []
        self.movements = []
        self._isRotated = False if self._orientation is not None else True

        self._formatImage(image)
        
        if (detectFaces is None and self._detectFaces) or detectFaces:
            self.face_detection()

        if (detectObjects is None and self._detectObjects) or detectObjects:
            self.object_detection()

        if (detectMotion is None and self._detectMotion) or detectMotion:
            self.motion_detection()

        end = time.time()
        
        self.rtSecs = end - start
        self.rt_logger.debug("Executed in " + str(self.rtSecs) + " seconds")

    def face_detection(self, image=None):
        self._formatImage(image)
        self.faces = []

        face_locations = face_recognition.face_locations(self._scaledRGBImage, model=self._faceModel)
        face_names = None
        if self._recognizeFaces:
            face_encodings = face_recognition.face_encodings(self._scaledRGBImage, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(self._faceEncodings, face_encoding)
                name = self._defaultFaceName

                face_distances = face_recognition.face_distance(self._faceEncodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self._faceNames[best_match_index]

                face_names.append(name)

        for idx, (stop, sright, sbottom, sleft) in enumerate(face_locations):
            left = int(sleft * self._faceScaleUpFactor)
            top = int(stop * self._faceScaleUpFactor)
            right = int(sright * self._faceScaleUpFactor)
            bottom = int(sbottom * self._faceScaleUpFactor)

            face_name = face_names[idx] if self._recognizeFaces else ""

            if self._imageMarkup:
                cv2.rectangle(self.image, (left, top), (right, bottom), self._faceOutlineColor, 2)

                if self._recognizeFaces and self._faceShowNames:
                    cv2.rectangle(self.image, (left, bottom - 18), (right, bottom), self._faceOutlineColor, cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(self.image, face_names[idx] if face_names is not None else "", (left + 6, bottom - 6), font, 0.5, self._faceFontColor, 1)

            if isinstance(self.faces, list):
                self.faces.append({ 
                    "type": "Face", 
                    "confidence": 1.0, 
                    "name": face_name, 
                    "location": { 
                        "left": left, 
                        "top": top, 
                        "right": right, 
                        "bottom": bottom 
                    },
                    "scaled_location": { 
                        "left": sleft, 
                        "top": stop, 
                        "right": sright, 
                        "bottom": sbottom 
                    } 
                })

    def object_detection(self, image=None):
        self._formatImage(image)

        self.objects = []

        classIndex, confidence, bbox = self._objModel.detect(self._scaledBGRImage, confThreshold=0.5)
        font = cv2.FONT_HERSHEY_PLAIN

        try:
            for classInd, conf, boxes in zip(classIndex.flatten(), confidence.flatten(), bbox):
                sleft = boxes[0]
                stop = boxes[1]
                sright = boxes[0] + boxes[2]
                sbottom = boxes[1] + boxes[3]

                left = int(sleft * self._faceScaleUpFactor)
                top = int(stop * self._faceScaleUpFactor)
                right = int(sright * self._faceScaleUpFactor)
                bottom = int(sbottom * self._faceScaleUpFactor)

                class_name = None

                if classInd > 0 and classInd <= len(self._objLabels):
                    class_name = self._objLabels[classInd - 1]

                if isinstance(self.objects, list):
                    self.objects.append({ 
                        "type": "Object", 
                        "confidence": conf, 
                        "name": class_name,
                        "location": { 
                            "left": left, 
                            "top": top, 
                            "right": right, 
                            "bottom": bottom 
                        },
                        "scaled_location": { 
                            "left": sleft, 
                            "top": stop, 
                            "right": sright, 
                            "bottom": sbottom 
                        } 
                    })

                if self._imageMarkup:
                    cv2.rectangle(self.image, (left, top), (right, bottom), self._objOutlineColor, 2)
                    if class_name is not None and self._objShowNames:
                        cv2.rectangle(self.image, (left, bottom - 18), (right, bottom), self._objOutlineColor, cv2.FILLED)
                        font = cv2.FONT_HERSHEY_DUPLEX
                        cv2.putText(self.image, class_name, (left + 6, bottom - 6), font, 0.5, self._objFontColor, 1)

        except AttributeError:
            pass

    def motion_detection(self, image=None):
        self._formatImage(image)

        self.movements = []

        self._scaledBWImage = cv2.cvtColor(self._scaledBGRImage, cv2.COLOR_BGR2GRAY)
        self._scaledBWImage = cv2.GaussianBlur(src=self._scaledBWImage, ksize=(5, 5), sigmaX=0)

        if self._lastScaledBWImage is None:
            return
        
        diff_frame = cv2.absdiff(src1=self._scaledBWImage, src2=self._lastScaledBWImage)

        kernel = np.ones((5, 5))
        diff_frame = cv2.dilate(diff_frame, kernel, 1)

        thresh_frame = cv2.threshold(src=diff_frame, thresh=self._motionThreshold, maxval=255, type=cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(image=thresh_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < self._motionMinArea:
                # too small: skip!
                continue

            (x, y, w, h) = cv2.boundingRect(contour)

            if isinstance(self.movements, list):
                self.movements.append({ 
                    "type": "Movement", 
                    "confidence": 1.0, 
                    "location": { 
                        "left": int(float(x) * self._faceScaleUpFactor), 
                        "top": int(float(y) * self._faceScaleUpFactor), 
                        "right": int(float(x + w) * self._faceScaleUpFactor), 
                        "bottom": int(float(y + h) * self._faceScaleUpFactor) 
                    },
                    "scaled_location": { 
                        "left": int(x), 
                        "top": int(y), 
                        "right": int(x + w), 
                        "bottom": int(y + h)
                    } 
                })

            if self._imageMarkup:
                cv2.rectangle(img=self.image, 
                              pt1=(int(float(x) * self._faceScaleUpFactor), int(float(y) * self._faceScaleUpFactor)), 
                              pt2=(int(float(x) * self._faceScaleUpFactor) + int(float(w) * self._faceScaleUpFactor), 
                                   int(float(y) * self._faceScaleUpFactor) + int(float(h) * self._faceScaleUpFactor)), 
                              color=self._motionOutlineColor, 
                              thickness=2)