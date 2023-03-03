import time
import cv2
import mediapipe as mp


cap = cv2.VideoCapture("face_videos/vid2.mp4")


class FPS:
    def __init__(self) -> None:
        self.pTime = 0
        self.cTime = 0
    def show(self,img):
        self.cTime = time.time() # current time  
        fps = 1 / (self.cTime - self.pTime)
        self.pTime = self.cTime
        cv2.putText(img, "fps:"+ str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),3)

fps = FPS()

mpFaceDetection = mp.solutions.face_detection
# mpFaceDetection = mp.solutions.mediapipe.python.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
# mpDraw = mp.solutions.mediapipe.python.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()


while True:
    success, img = cap.read()


    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    print(results)
    if results.detections:
        for id, detection in enumerate(results.detections):
            mpDraw.draw_detection(img,detection)
            print(id,detection)
            print(detection.score)
            print(detection.location_data.relative_bounding_box)

            bboxC = detection.location_data.relative_bounding_box # bounding box
            ih, iw, ic = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                    int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(img,bbox,(255,0,255),2)
            cv2.putText(img, f'{int(detection.score[0]*100)}%', 
                        (bbox[0],bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                        2, (0,255,0),2)


    fps.show(img)
    cv2.imshow("Image", img)
    cv2.waitKey(1)