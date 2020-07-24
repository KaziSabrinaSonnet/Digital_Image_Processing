
#%%
#importing cv2
import cv2
# Using cv2.imread() method
img1 = cv2.imread('3_1.bmp')
# Displaying the image using cv2.imshow()
cv2.imshow('3_1.bmp', img1)
#Maintain output window until user presses a key
cv2.waitKey(0)
cv2.destroyAllWindows()
#%%
Blue_Band = img1[:, :, 0]
Green_Band = img1[:, :, 1]
Red_Band = img1[:, :, 2]

# %%
BB_rows = Blue_Band.shape[0]
GB_rows = Green_Band.shape[0]
RB_rows = Red_Band.shape[0]
BB_coloumns = Blue_Band.shape[1]
GB_coloumns = Green_Band.shape[1]
RB_coloumns = Red_Band.shape[1]

# %%
#Algorithm
new_BB= np.zeros((int(BB_rows), int(BB_coloumns)))
#gamma = 3
gama_correction = 1.3
for i in range(0, BB_rows):
    for j in range(0, BB_coloumns):
        new_BB[i, j] = 255*(Blue_Band[i, j]/255)**gama_correction

Fin_BB = np.uint8(new_BB)
#%%
new_GB= np.zeros((int(GB_rows), int(GB_coloumns)))
#gamma = 3
gama_correction = 0.7
for i in range(0, GB_rows):
    for j in range(0, GB_coloumns):
        new_GB[i, j] = 255*(Green_Band[i, j]/255)**gama_correction

Fin_GB = np.uint8(new_GB)
#%%
new_RB= np.zeros((int(RB_rows), int(RB_coloumns)))
#gamma = 3
gama_correction = 0.9
for i in range(0, RB_rows):
    for j in range(0, RB_coloumns):
        new_RB[i, j] = 255*(Red_Band[i, j]/255)**gama_correction

Fin_RB = np.uint8(new_RB)
#%%
image_final = cv2.merge((Fin_BB, Fin_GB, Fin_RB))
cv2.imshow('Natural Image', image_final)
#Maintain output window until user presses a key
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
