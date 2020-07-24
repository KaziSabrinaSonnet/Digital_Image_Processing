#%%
import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('5_1.bmp',0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
Magnitude = 20*np.log(np.abs(fshift))
Phase = np.angle(fshift)
#%%
plt.imshow(img, cmap = 'gray')
plt.title('Input Image')
plt.xticks([])
plt.yticks([])
#%%
from mpldatacursor import datacursor
plt.imshow(Magnitude, cmap = 'gray')
plt.title('Magnitude')
plt.xticks([])
plt.yticks([])
datacursor(display='single')
plt.show()
#%%
plt.imshow(Phase, cmap = 'gray')
plt.title('Phase')
plt.xticks([])
plt.yticks([])
plt.show()
#%%
fshift[382, 382]=0
fshift[382, 383]=0
fshift[382, 384]=0
fshift[383, 382]=0
fshift[383, 383]=0
fshift[383, 384]=0
fshift[384, 382]=0
fshift[384, 383]=0
fshift[384, 384]=0
fshift[384, 385]=0
fshift[384, 386]=0
fshift[385, 384]=0
fshift[385, 385]=0
fshift[385, 386]=0
fshift[386, 384]=0
fshift[386, 385]=0
fshift[386, 386]=0
#%%
fshift[126, 126]= 0
fshift[126, 127]= 0
fshift[126, 128]= 0
fshift[127, 126]= 0
fshift[127, 127]= 0
fshift[127, 128]= 0
fshift[128, 126]= 0
fshift[128, 127]= 0
fshift[128, 128]= 0
fshift[128, 129]= 0
fshift[128, 130]= 0
fshift[129, 128]= 0
fshift[129, 129]= 0
fshift[129, 130]= 0
fshift[130, 128]= 0
fshift[130, 129]= 0
fshift[130, 130]= 0

#%%
fshift[384, 126]= 0
fshift[384, 127]= 0
fshift[384, 128]= 0
fshift[385, 126]= 0
fshift[385, 127]= 0
fshift[385, 128]= 0
fshift[386, 126]= 0
fshift[386, 127]= 0
fshift[386, 128]= 0
fshift[386, 129]= 0
fshift[386, 130]= 0
fshift[387, 128]= 0
fshift[387, 129]= 0
fshift[387, 130]= 0
fshift[388, 128]= 0
fshift[388, 129]= 0
fshift[388, 130]= 0
#%%
fshift[126, 382]= 0
fshift[126, 383]= 0
fshift[126, 384]= 0
fshift[127, 382]= 0
fshift[127, 383]= 0
fshift[127, 384]= 0
fshift[128, 382]= 0
fshift[128, 383]= 0
fshift[128, 384]= 0
fshift[128, 385]= 0
fshift[128, 386]= 0
fshift[129, 384]= 0
fshift[129, 385]= 0
fshift[129, 386]= 0
fshift[130, 384]= 0
fshift[130, 385]= 0
fshift[130, 386]= 0
#%%
#Denoisying
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)
plt.imshow(img_back, cmap = 'gray')
# %%
#Denoisying image's magnitude and phase 
f1 = np.fft.fft2(img_back)
fshift1 = np.fft.fftshift(f1)
Magnitude1 = 20*np.log(np.abs(fshift1))
Phase1 = np.angle(fshift1)
from mpldatacursor import datacursor
plt.imshow(Magnitude1, cmap = 'gray')
plt.title('Denoised Image Magnitude')
plt.xticks([])
plt.yticks([])
datacursor(display='single')
plt.show()
plt.imshow(Phase1, cmap = 'gray')
plt.title('Denoised Image Phase')
plt.xticks([])
plt.yticks([])
plt.show()
# %%
