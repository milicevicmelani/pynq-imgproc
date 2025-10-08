
# coding: utf-8

# In[ ]:


from pynq import Overlay, Xlnk
from pynq.overlays.base import BaseOverlay
from pynq.lib.video import *
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import time


# In[ ]:


base = BaseOverlay("base.bit")
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


overlay = Overlay("imageProcessingSystem.bit")
overlay.download()

dma = overlay.axi_dma_0


# In[ ]:


xlnk=Xlnk()
img = cv2.imread("lena_gray.bmp")
resize = cv2.resize(img, (512, 512))
gray=cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
height, width = gray.shape
size=height*width

send_buf = xlnk.cma_array(shape=(size,), dtype=np.uint8)
recv_buf = xlnk.cma_array(shape=(size,), dtype=np.uint8)

np.copyto(send_buf,np.zeros_like(send_buf))
np.copyto(recv_buf,np.zeros_like(recv_buf))


# In[ ]:


np.copyto(send_buf,gray.reshape(-1))
dma.sendchannel.transfer(send_buf)
dma.recvchannel.transfer(recv_buf)
dma.sendchannel.wait()
print(recv_buf)


# In[ ]:


sobel=np.empty_like(gray)
np.copyto(sobel,recv_buf.reshape(gray.shape))
recv_buf.close()
send_buf.close()

plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(img, cmap='gray')

plt.subplot(1,2,2)
plt.title("Processed in FPGA")
plt.imshow(sobel, cmap='gray')
plt.show()


# In[ ]:


dma.recvchannel.stop()
dma.sendchannel.stop()


del dma.sendchannel
del dma.recvchannel
del dma

