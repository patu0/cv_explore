import cv2
import mediapipe as mp
import time


class FPS:
    def __init__(self) -> None:
        self.pTime = 0
        self.cTime = 0
    def show(self,img):
        self.cTime = time.time() # current time  
        fps = 1 / (self.cTime - self.pTime)
        self.pTime = self.cTime
        # return cv2.putText(img, str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        cv2.putText(img, str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

cap = cv2.VideoCapture(fr'D:\Code\cv\stock_videos\vid2.mp4')
fps = FPS()

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS,
                            mpDraw.DrawingSpec(color=(0,0,255), thickness=10, circle_radius=2),
                            mpDraw.DrawingSpec(color=(0,255,255), thickness=5)
                            )
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            print(id, lm)
            cx, cy = int(lm.x * w), int(lm.y * h)
            # cv2.circle(img, (cx, cy), 20, (255, 0, 0), cv2.FILLED)
    img = cv2.resize(img, (700, 500))
    fps.show(img)

    cv2.imshow("Image", img)
    cv2.waitKey(10)
