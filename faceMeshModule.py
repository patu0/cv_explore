import cv2 
import mediapipe as mp
import time
 
class FaceMeshDetector():
    def __init__(self,
               staticImageMode=False,
               maxNumOfFaces=2,
               refineLandmarks=False,
               minDetectionConfidence=0.5,
               minTrackingConfidence =0.5):
        self.staticImageMode = staticImageMode
        self.maxNumOfFaces = maxNumOfFaces
        self.refineLandmarks = refineLandmarks
        self.minDetectionConfidence = minDetectionConfidence
        self.minTrackingConfidence = minTrackingConfidence 

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(max_num_faces=maxNumOfFaces)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1) 

    def drawMesh(self,img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(imgRGB)
        if results.multi_face_landmarks:
            #draw mesh over face
            for faceLms in results.multi_face_landmarks:
                print("Confidence:", faceLms.score)
                # print("+++++++")
                # print("faceLms:", faceLms)
                self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                self.drawSpec,self.drawSpec)
                # extract info from current faceLms
                for id,lm in enumerate(faceLms.landmark):
                    # print("==============")
                    # print("id:",id,"lm:",lm)
                    ih, iw, ic = img.shape
                    x,y = int(lm.x*iw), int(lm.y*ih)
                    print(id,x,y)
        return img
def main():
    cap = cv2.VideoCapture(fr"D:\Code\cv_explore\face_videos\vid2.mp4")
    pTime = 0
    detector = FaceMeshDetector()
    while True:
        success, img = cap.read()
        img = detector.drawMesh(img)


        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
        3, (255, 0, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()