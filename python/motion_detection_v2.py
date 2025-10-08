
# coding: utf-8

# In[2]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import time


# In[3]:


#Overlay učitavanje
from pynq.overlays.base import BaseOverlay
from pynq.lib.video import *

base = BaseOverlay("base.bit")


# In[4]:


# Konfiguracija monitora/izlaza
Mode = VideoMode(640, 480, 24)
hdmi_out = base.video.hdmi_out
hdmi_out.configure(Mode,PIXEL_BGR)
hdmi_out.start()

# Konfiguracija kamere/ulaza
frame_in_w = 640
frame_in_h = 480


# In[5]:


videoIn = cv2.VideoCapture(0)
videoIn.set(cv2.CAP_PROP_FRAME_WIDTH, frame_in_w);
videoIn.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_in_h);
print("capture device is open: " + str(videoIn.isOpened())) ## Check the webcam is available


# In[6]:


#Funkcija za pronalazak kontura

#def getContours(frame):
    #contour_data = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #contours = contour_data[0] if len(contour_data) == 2 else contour_data[1]
    #return contours

def find_ROI(frame, contours, min_circularity=0.7):
    if not contours:
        return None

    potential_basket = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(potential_basket)
    center = (int(x), int(y))
    radius = int(radius)

    circle_area = 3.14159 * (radius ** 2)
    circularity = cv2.contourArea(potential_basket) / circle_area

    height, width = frame.shape[:2]
    if circularity > min_circularity and x - radius >= 0 and y - radius >= 0 and x + radius <= width and y + radius <= height:
        cv2.circle(frame, center, radius, (0, 255, 0), 3)
        cv2.circle(frame, center, 2, (0, 0, 255), 3)
        return potential_basket

    return None


# In[8]:


fgbg = cv2.createBackgroundSubtractorMOG2()
score = 0
ROI = None
basket_found = False
frame_count = 0       

# Blob detector za tamnu lopticu
params = cv2.SimpleBlobDetector_Params()
params.filterByColor = True
params.blobColor = 0         
params.filterByArea = True
params.minArea = 20
params.maxArea = 100
detector = cv2.SimpleBlobDetector_create(params)
insideBlobs=[]
while not base.buttons[0].read():
    ret, frame = videoIn.read()
    if not ret:
        break

    frame_count += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    if not basket_found:
        edges = cv2.Canny(blur, 100, 200)
        contour_data = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contour_data[0] if len(contour_data) == 2 else contour_data[1]
        basket_candidate = find_ROI(frame, contours)
        if basket_candidate is not None:
            ROI = basket_candidate
            basket_found = True

    if basket_found:
        (x, y), radius = cv2.minEnclosingCircle(ROI)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(frame, center, radius, (0, 255, 0), 3)
        cv2.circle(frame, center, 2, (0, 0, 255), 3)

        bx, by, bw, bh = cv2.boundingRect(ROI)
        roi_frame = frame[by:by+bh, bx:bx+bw]

        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        
        fgmask = fgbg.apply(blur)               
        fgmask = cv2.medianBlur(fgmask, 5)      

        inv_blur = cv2.bitwise_not(blur)
        motionROI = cv2.bitwise_and(inv_blur, fgmask)  
       
        keypoints = detector.detect(motionROI)

        margin=10
        newBlobs=[]
        
        if  keypoints and frame_count % 5 == 0:
            newBlobs.clear()
            for kp in keypoints:
                cx, cy = int(kp.pt[0]), int(kp.pt[1]) 
                global_x, global_y = bx + cx, by + cy
                print("keypoint",global_x, global_y)
                if insideBlobs:
                    for blob in insideBlobs:
                        alreadyIn=blob[0] <= global_x <= blob[0]+2*margin and  blob[1] <= global_y <= blob[1]+2*margin
                        if alreadyIn:
                            #("1Blob unutra",blob)
                            break
                    if not alreadyIn:
                        newBlobs.append((global_x, global_y))
                        #print("2novi blob",(global_x, global_y))
                        continue
                            
                else:
                    newBlobs.append((global_x, global_y))
                    #print("3Blob novi:",global_x, global_y)
                    break 

            for nBlob in newBlobs:
                if(bx <= nBlob[0] <= bx + bw and by <= nBlob[1] <= by + bh):   
                    insideBlobs.append((nBlob[0]-margin, nBlob[1]-margin))
                    score+=1
                    #print("4Novi blob, score +1:", nBlob)
            #print("InsideBlobs:",  insideBlobs)
            
                

        cv2.putText(frame, f"{score}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)

        # Pošalji frame na HDMI
        outframe = hdmi_out.newframe()
        outframe[0:480, 0:640, :] = frame[0:480, 0:640, :]
        hdmi_out.writeframe(outframe)


# In[ ]:


videoIn.release()
hdmi_out.close()
del hdmi_out

