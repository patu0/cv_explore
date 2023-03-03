import cv2
import time
import PoseModule as pm

cap = cv2.VideoCapture(fr'D:\Code\cv\stock_videos\vid3.mp4')
fps = pm.FPS()
detector = pm.poseDetector()
while True:
    success, img = cap.read()
    img = detector.findPose(img)
    img = cv2.resize(img, (700, 500))
    lmList = detector.findPosition(img)
    if len(lmList) != 0:
        print(lmList[14])
        cv2.circle(img, (lmList[14][1],lmList[14][2]), 20, (0, 0, 255), cv2.FILLED)
    fps.show(img)
    cv2.imshow("Image", img)
    cv2.waitKey(10)