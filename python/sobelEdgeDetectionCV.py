
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
from pynq.overlays.base import BaseOverlay
from pynq.lib.video import *
import time

base = BaseOverlay("base.bit")


# In[ ]:


Mode = VideoMode(640,480,8)
hdmi_out = base.video.hdmi_out
hdmi_out.configure(Mode,PIXEL_GRAY)
hdmi_out.start()

# camera (input) configuration
frame_in_w = 640
frame_in_h = 480

videoIn = cv2.VideoCapture(0)
videoIn.set(cv2.CAP_PROP_FRAME_WIDTH, frame_in_w);
videoIn.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_in_h);
print("capture device is open: " + str(videoIn.isOpened())) ## Check the webcam is available


# In[ ]:


ret, frame_vga = videoIn.read()

if (ret):
    outframe = hdmi_out.newframe()
    outframe[:] = frame_vga
    hdmi_out.writeframe(outframe)
else:
    raise RuntimeError("Error while reading from camera.")


# In[ ]:


start = time.time()
while not base.buttons[0].read():
    ret, frame_webcam = videoIn.read()
    
    if ret:
        gray = cv2.cvtColor(frame_webcam, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        sobel = np.sqrt(sobelx**2 + sobely**2)
        sobel = cv2.convertScaleAbs(sobel)
        threshold = 200  
        edges = np.where(sobel > threshold, 255, 0).astype(np.uint8)
        outframe = hdmi_out.newframe()
        np.copyto(outframe, edges)
        hdmi_out.writeframe(outframe)
    else:
        readError += 1

end = time.time()


# In[ ]:


videoIn.release()
hdmi_out.close()
del hdmi_out

