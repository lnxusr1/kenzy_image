from . import core
import cv2

detector = core.detector(detectFaces=False, detectMotion=False, detectObjects=True, imageMarkup=True, scaleFactor=0.5)
# detector.addFace("/home/lnxusr1/Pictures/faces/lnxusr1/IMG_5421.jpg", "LNXUSR1")

cam = cv2.VideoCapture(0)
while True:

    ret, frame = cam.read()
    detector.analyze(frame)

    cv2.imshow('Motion detector', detector.image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()