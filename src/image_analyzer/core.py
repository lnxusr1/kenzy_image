import os
import face_recognition
import cv2
import numpy as np
import time


class detector(object):
    def __init__(self, detectFaces=False, detectObjects=False, detectMotion=False, scaleFactor=1.0, imageMarkup=False):

        self._imageMarkup = imageMarkup
        self._faceNames = []
        self._faceEncodings = []
        
        self._detectFaces = detectFaces
        self._recognizeFaces = False
        self._detectObjects = detectObjects
        self._detectMotion = detectMotion

        self._faceScaleDownFactor = scaleFactor  # (1.0 >= VALUE > 0)
        self._faceModel = "hog"  # hog or cnn
        self._faceScaleUpFactor = 1.0  
        self._defaultFaceName = "Unknown"
        
        if self._faceScaleDownFactor < 1 and self._faceScaleDownFactor > 0:
            self._faceScaleUpFactor = ((1.0 - float(self._faceScaleDownFactor)) / float(self._faceScaleDownFactor)) + 1.0

        self.image = None
        self._lastImage = None
        self._scaledBGRImage = None
        self._scaledRGBImage = None
        self._scaledBWImage = None
        self._lastScaledBWImage = None

        self._objConfigFile = os.path.join(os.path.dirname(__file__), "resources", "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
        self._objModelFile = os.path.join(os.path.dirname(__file__), "resources", "frozen_inference_graph.pb")
        self._objLabelFile = os.path.join(os.path.dirname(__file__), "resources", "labels.txt")

        self._objModel = cv2.dnn_DetectionModel(self._objModelFile, self._objConfigFile)
        self._objModel.setInputSize(320, 320)  # greater this value the better the results; tune it for best output
        self._objModel.setInputScale(1.0 / 127.5)
        self._objModel.setInputMean((127.5, 127.5, 127.5))
        self._objModel.setInputSwapRB(True)
        self._objLabels = []
        
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

        self._detectFaces = True
        self._recognizeFaces = True

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

        self._scaledBGRImage = None
        self._scaleImage()

        self._scaledBWImage = None
        self._bwImage()

    def _scaleImage(self):
        if self._scaledBGRImage is None:
            if self._faceScaleDownFactor != 1.0:
                self._scaledBGRImage = cv2.resize(self.image, (0, 0), fx=self._faceScaleDownFactor, fy=self._faceScaleDownFactor, interpolation=cv2.INTER_AREA)
            else:
                self._scaledBGRImage = self.image

            self._scaledRGBImage = cv2.cvtColor(self._scaledBGRImage, cv2.COLOR_BGR2RGB)
    
    def _bwImage(self):
        self._scaleImage()
        if self._scaledBWImage is None:
            self._scaledBWImage = cv2.cvtColor(self._scaledBGRImage, cv2.COLOR_BGR2GRAY)
            self._scaledBWImage = cv2.GaussianBlur(src=self._scaledBWImage, ksize=(5, 5), sigmaX=0)

    def _loadLabels(self):
        with open(self._objLabelFile, 'rt') as fp:
            self._objLabels = fp.read().rstrip('\n').split('\n')

    def analyze(self, image):
        start = time.time()

        self.faces = []
        self.objects = []
        self.movements = []

        self._formatImage(image)
        
        if self._detectFaces:
            self.face_detection()

        if self._detectObjects:
            self.object_detection()

        if self._detectMotion:
            self.motion_detection()

        end = time.time()

        print("Executed in", (end - start), "seconds.")

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
                cv2.rectangle(self.image, (left, top), (right, bottom), (0, 0, 255), 2)

                if self._recognizeFaces:
                    cv2.rectangle(self.image, (left, bottom - 18), (right, bottom), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(self.image, face_names[idx] if face_names is not None else "", (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

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

        print(self.faces)

    def object_detection(self, image=None):
        self._formatImage(image)

        self.objects = []

        classIndex, confidence, bbox = self._objModel.detect(self._scaledBGRImage, confThreshold=0.5)
        font = cv2.FONT_HERSHEY_PLAIN

        print(classIndex)

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
                    cv2.rectangle(self.image, (left, top), (right, bottom), (255, 0, 0), 2)
                    if class_name is not None:
                        cv2.rectangle(self.image, (left, bottom - 18), (right, bottom), (255, 0, 0), cv2.FILLED)
                        font = cv2.FONT_HERSHEY_DUPLEX
                        cv2.putText(self.image, class_name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        except AttributeError:
            pass

        print(self.objects)

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

        thresh_frame = cv2.threshold(src=diff_frame, thresh=20, maxval=255, type=cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(image=thresh_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 50:
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
                              color=(0, 255, 0), 
                              thickness=2)
        
        print(self.movements)