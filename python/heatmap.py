
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from pynq.overlays.base import BaseOverlay
from pynq.lib.video import *

base = BaseOverlay("base.bit")


# In[2]:


# Konfiguracija monitora/izlaza
Mode = VideoMode(640, 480, 24)
hdmi_out = base.video.hdmi_out
hdmi_out.configure(Mode,PIXEL_BGR)
hdmi_out.start()

# Konfiguracija kamere/ulaza
frame_in_w = 640
frame_in_h = 480


# In[3]:


videoIn = cv2.VideoCapture(0)
videoIn.set(cv2.CAP_PROP_FRAME_WIDTH, frame_in_w);
videoIn.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_in_h);
print("capture device is open: " + str(videoIn.isOpened())) ## Check the webcam is available


# In[4]:


fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

ret, frame = videoIn.read()
height, width = frame.shape[:2]

heatmap = np.zeros((height, width), dtype=np.float32)

while not base.buttons[0].read():
    ret, frame = videoIn.read()
    if not ret:
        break

    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    heatmap += fgmask.astype(np.float32) / 255 
    heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_color = cv2.applyColorMap(heatmap_norm.astype(np.uint8), cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)

    outframe = hdmi_out.newframe()
    outframe[0:480, 0:640, :] = overlay[0:480, 0:640, :]
    hdmi_out.writeframe(outframe)


# In[5]:


heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
heatmap_color = cv2.applyColorMap(heatmap_norm.astype(np.uint8), cv2.COLORMAP_JET)
cv2.imwrite("motion_heatmap.png", heatmap_color)


# In[ ]:


videoIn.release()
hdmi_out.close()
del hdmi_out

