# Control Volume of Computer with Hand
import cv2
import time 
import numpy as np
import HandTrackingModule as htm

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# volume.GetMute()
# volume.GetMasterVolumeLevel()
# print(volume.GetVolumeRange())
# volume.SetMasterVolumeLevel(0.0, None) # 100%
# volume.SetMasterVolumeLevel(-5.0, None) # 
# volume.SetMasterVolumeLevel(-20.0, None) # 26%
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]



def calcFingerTipDistance(x1,y1,x2,y2):
    distance = (((x2-x1)**2) + ((y2-y1)**2))**0.5
    return distance

cap = cv2.VideoCapture(0)
wCam, hCam = 1280,720
cap.set(3,wCam)
cap.set(4,hCam)

# cTime = 0
# pTime = 0

detector = htm.handDetector(
    maxHands=1,
    detectConfi=0.5
    )
fps = htm.FPS()
vol = 0
# bar_fill = 100
bar_fill = np.interp(volume.GetMasterVolumeLevel(),[minVol,maxVol],[400,100])
volPer = np.interp(volume.GetMasterVolumeLevel(),[minVol,maxVol],[0,100])
print("initial bar fill:", bar_fill)
while True:
    success, img = cap.read()

    img = detector.findHands(img)
    img = fps.show(img)
    lmList = detector.findPosition(img,draw=False)
    if len(lmList) != 0:
        finger1_id = 4
        finger2_id = 8
        finger3_id = 12
        finger4_id = 16
        finger5_id = 20
        # print(lmList[finger1_id], lmList[finger2_id])
    
        x1, y1 = lmList[finger1_id][1], lmList[finger1_id][2]
        x2, y2 = lmList[finger2_id][1], lmList[finger2_id][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img,(x1,y1),10,(0,255,0))
        cv2.circle(img,(x2,y2),10,(0,255,0))
        cv2.line(img, (x1,y1),(x2,y2),(0,0,255),3)
        cv2.circle(img,(cx,cy),10,(255,0,0),cv2.FILLED)
        length = calcFingerTipDistance(x1,y1,x2,y2)
        
        vol = np.interp(length,[50,300],[minVol,maxVol])
        color = np.interp(length,[50,300],[0,255])
        bar_fill = np.interp(length,[50,300],[400,100])
        volPer = np.interp(length,[50,300], [0,100])
        print("len",int(length), "vol", vol)
        volume.SetMasterVolumeLevel(vol, None)
        cv2.circle(img,(cx,cy),10,(0,color,0),cv2.FILLED)

    cv2.rectangle(img,(50,100),(85,400),(255,255,255),3)
    cv2.rectangle(img,(50,int(bar_fill)),(85,400),(0,0,0),cv2.FILLED)
    cv2.putText(img, f'{int(volPer)} %',(40,450), cv2.FONT_HERSHEY_PLAIN,3,
                (0,0,255),3)


        # if length < 50:
        #     cv2.circle(img,(cx,cy),10,(0,255,0),cv2.FILLED)
        #     # volume.SetMasterVolumeLevel(minVol, None) 
        # elif length > 50 and length < 200:
        #     # volume.SetMasterVolumeLevel((minVol - maxVol)/2, None) 
        #     cv2.circle(img,(cx,cy),10,(0,100,0),cv2.FILLED)
        # else:
        #     pass
            # volume.SetMasterVolumeLevel(maxVol, None)     
        # x3, y3 = lmList[finger3_id][1], lmList[finger3_id][2]
        # x4, y4 = lmList[finger4_id][1], lmList[finger4_id][2]
        # x5, y5 = lmList[finger5_id][1], lmList[finger5_id][2]

        # cv2.circle(img,(x3,y3),10,(0,255,0))
        # cv2.circle(img,(x4,y4),10,(0,255,0))
        # cv2.circle(img,(x5,y5),10,(0,255,0))
        # cv2.line(img, (x1,y1),(x3,y3),(0,0,255),3)
        # cv2.line(img, (x1,y1),(x4,y4),(0,0,255),3)
        # cv2.line(img, (x1,y1),(x5,y5),(0,0,255),3)

    cv2.imshow("Img",img)
    cv2.waitKey(1)