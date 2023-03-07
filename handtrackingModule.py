import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self,
                mode=False,
                maxHands=2,
                complexity=1,
                detectConfi=0.5,
                trackConfi=0.5):
        
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detectConfi = detectConfi
        self.trackConfi = trackConfi
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.complexity,self.detectConfi,self.trackConfi) 
        self.mpDraw = mp.solutions.drawing_utils
    
    def findHands(self, img, draw=True):
        # input rgb 
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            #extract method of each hand
            for handLMs in self.results.multi_hand_landmarks:
                # get info from each hand
                if draw:
                    self.mpDraw.draw_landmarks(img, handLMs, 
                                               self.mpHands.HAND_CONNECTIONS)
        return img
    
    def findPosition(self, img, handNum=0,draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNum]
            for id, lm in enumerate(myHand.landmark):
                #convert to pixels of screen
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h) # center x and y
                lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img, (cx,cy), 15, (255,0,255), cv2.FILLED)
        return lmList

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
        return img
    
def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img, handNum=0, draw=False)
        if len(lmList) != 0:
            print(lmList[4])

        # Calculate & display FPS
        cTime = time.time() # current time
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN,3,
                    (255,0,255),3)
        
        cv2.imshow("Image", img)
        cv2.waitKey(1) 

if __name__ == "__main__":
    main()