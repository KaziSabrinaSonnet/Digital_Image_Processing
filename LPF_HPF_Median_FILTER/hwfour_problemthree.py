#%%
#Problem: 03
#importing cv2
import cv2
# Using cv2.imread() method
img = cv2.imread('4_2.bmp')
# Displaying the image using cv2.imshow()
cv2.imshow('Original Image', img)
#Maintain output window until user presses a key
cv2.waitKey(0)
cv2.destroyAllWindows()
#%%
import numpy as np
image_first_band = img[:,:,0]
rows = image_first_band.shape[0]
coloumns= image_first_band.shape[1] 
#%%
#High PASS FILTER 
#padding
import math
new_row= rows+2
new_coloumn= coloumns+2
padded_matrix = np.zeros((int(new_row), int(new_coloumn)))
s= -1
for i in range(1, new_row-1):
    s= s+1 
    t= 0
    for j in range(1, new_coloumn-1):
        padded_matrix[i, j]= image_first_band[s,t]
        t= t+1
#%%
p= 0
for q in range(1, new_coloumn-1):
    padded_matrix[p, q]= padded_matrix[p+1, q ]

u= new_row-1
for v in range(1, new_coloumn-1):
    padded_matrix[u, v]= padded_matrix[u-1, v ]
#%%
"""
padded_matrix[0, 0]= padded_matrix[0, 1]
padded_matrix[new_row-1, 0]= padded_matrix[new_row-1, 1]
padded_matrix[0, new_row-1] = padded_matrix[0, new_row-2]
padded_matrix[new_row-1, new_row-1]= padded_matrix[new_row-1, new_row-2]
"""
#%%
b= 0
for a in range(0, 482): 
    padded_matrix[a, b]= padded_matrix[a, b+1]

d= 641
for c in range(0, 482): 
    padded_matrix[c, d]= padded_matrix[c, d-1]

HPF_matrix= np.zeros((int(rows), int(coloumns)))

for i in range(0, rows): 
    for j in range(0, coloumns):
        HPF_matrix[i, j]= math.floor((padded_matrix[i, j]*1+padded_matrix[i, j+1]*1+padded_matrix[i, j+2]*1+
                          padded_matrix[i+1, j]*1+padded_matrix[i+1, j+1]*(-8)+padded_matrix[i+1, j+2]*1+
                          padded_matrix[i+2, j]*1+padded_matrix[i+2, j+1]*1+padded_matrix[i+2, j+2]*1)/9)
                          


HPFFiltered= np.uint8(HPF_matrix)
image_final4 = cv2.merge((HPFFiltered, HPFFiltered, HPFFiltered))
cv2.imshow('HPF_Filtered_Image', image_final4)
#Maintain output window until user presses a key
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
