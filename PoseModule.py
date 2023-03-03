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

class poseDetector():
    def __init__(self,
                 mode = False,
                 modelcomp = 1,
                 smooth = True,
                 enableSeg=False,
                 smoothSeg=True,
                 detectCon=0.5,
                 trackCon=0.5
                 ) -> None:
        
        self.mode = mode
        self.modelcomp = modelcomp
        self.smooth = smooth
        self.enableSeg = False
        self.smoothSeg = smoothSeg
        self.detectCon = detectCon
        self.trackCon = trackCon



        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
                                    self.mode,self.modelcomp,
                                    self.smooth,
                                    self.enableSeg,
                                    self.smoothSeg,
                                    self.detectCon,self.trackCon
                                    )

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS,
                                    self.mpDraw.DrawingSpec(color=(0,0,255), thickness=10, circle_radius=2),
                                    self.mpDraw.DrawingSpec(color=(0,255,255), thickness=5)        
                                    )
        return img
    
    def findPosition(self,img,draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList
def main():
    cap = cv2.VideoCapture(fr'D:\Code\cv\stock_videos\vid2.mp4')
    fps = FPS()
    detector = poseDetector()
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

if __name__ == "__main__":
    main()