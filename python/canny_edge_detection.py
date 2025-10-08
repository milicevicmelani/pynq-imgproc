
# coding: utf-8

# In[1]:


#Overlay
import cv2
from pynq.overlays.base import BaseOverlay
from pynq.lib.video import *

base = BaseOverlay("base.bit")


# In[2]:


# monitor configuration: 640*480 @ 60Hz
Mode = VideoMode(640,480,8)
hdmi_out = base.video.hdmi_out
hdmi_out.configure(Mode,PIXEL_GRAY)
hdmi_out.start()

# camera (input) configuration
frame_in_w = 640
frame_in_h = 480


# In[3]:


outframe = hdmi_out.newframe()
outframe[:] = [0, 0, 255]  # BGR: plava

hdmi_out.writeframe(outframe)

print("Plavi ekran bi trebao biti prikazan.")


# In[7]:


videoIn = cv2.VideoCapture(0)
videoIn.set(cv2.CAP_PROP_FRAME_WIDTH, frame_in_w);
videoIn.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_in_h);
print("capture device is open: " + str(videoIn.isOpened())) ## Check the webcam is available


# In[8]:


import numpy as np

ret, frame_vga = videoIn.read()

if (ret):
    outframe = hdmi_out.newframe()
    outframe[:] = frame_vga
    hdmi_out.writeframe(outframe)
else:
    raise RuntimeError("Error while reading from camera.")


# In[9]:


start = time.time()
lowThresh=100
highThresh=200
while not base.buttons[0].read():
    # read next image
    ret, frame_webcam = videoIn.read()
    if (ret):
        outframe = hdmi_out.newframe()
        cv2.Canny(frame_webcam, lowThresh, highThresh, edges=outframe)
        hdmi_out.writeframe(outframe)
    else:
        readError += 1
end = time.time()



# In[10]:


videoIn.release()

