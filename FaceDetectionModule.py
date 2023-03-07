import time
import cv2
import mediapipe as mp

class FPS:
    def __init__(self) -> None:
        self.pTime = 0
        self.cTime = 0
    def show(self,img):
        self.cTime = time.time() # current time  
        fps = 1 / (self.cTime - self.pTime)
        self.pTime = self.cTime
        cv2.putText(img, "fps:"+ str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),3)
        return img
class FaceDetector():
    def __init__(self, minDetectionCon=0.5,modelType=1):
        self.minDetectionCon = minDetectionCon
        self.modelType = modelType

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(min_detection_confidence=self.minDetectionCon, model_selection=self.modelType) 
    def findFaces(self,img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceDetection.process(imgRGB)
        # print(results)
        bboxs = []
        if results.detections:
            for id, detection in enumerate(results.detections):
                # self.mpDraw.draw_detection(img,detection) # this creates a default bounding box over detected face 
                # print(id,detection)
                # print(detection.score)
                # print(detection.location_data.relative_bounding_box)
                bboxC = detection.location_data.relative_bounding_box # bounding box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                        int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id,bbox,detection.score])
                
                if draw:
                    img = self.fancyDraw(img,bbox)

                    cv2.putText(img, f'{int(detection.score[0]*100)}%', 
                                (bbox[0],bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                                2, (0,255,0),2)
    
        return img, bboxs

    def fancyDraw(self, img, bbox, l=30, t=3, rt=1):
        """
        input:
        img
        single bbox
        """
        x, y, w, h = bbox # top left corner of bbox
        x1, y1 = x + w, y + h # bottom right corner bbox
        
        # cv2.rectangle(img,bbox,(0,255,0),rt)

        # top left corner: x,y
        cv2.line(img,(x,y), (x+l,y),(255,0,255),t) # top left corner toward right
        cv2.line(img,(x,y), (x,y+l),(255,0,255),t) # top left corner toward down
        # top right: x1, y
        cv2.line(img,(x1,y), (x1-l,y),(255,0,255),t) # top left corner toward left
        cv2.line(img,(x1,y), (x1,y+l),(255,0,255),t) # top left corner toward down
        # bottom left
        cv2.line(img,(x,y1), (x,y1-l),(255,0,255),t) # bottom left corner toward up
        cv2.line(img,(x,y1), (x+l,y1),(255,0,255),t) # bottom left corner toward right
        # bottom right
        cv2.line(img,(x1,y1), (x1,y1-l),(255,0,255),t) # bottom left corner toward up
        cv2.line(img,(x1,y1), (x1-l,y1),(255,0,255),t) # bottom left corner toward left

        return img

def main():
    fps = FPS()
    # cap = cv2.VideoCapture("face_videos/vid2.mp4")
    cap = cv2.VideoCapture(0)
    detector = FaceDetector()
    
    while True:
        success, img = cap.read()
    
        img,bboxs = detector.findFaces(img,True)
        print(bboxs)
        img = fps.show(img)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()