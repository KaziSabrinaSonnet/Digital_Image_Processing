#%%
#Problem: 01
#importing cv2
import cv2
# Using cv2.imread() method
img = cv2.imread('4_1.bmp')
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
s_p = np.random.randint(0, 256, size=(rows, coloumns))
# %%
for i in range(0, rows): 
    for j in range(0, coloumns):
        if s_p[i, j]==0 or s_p[i, j]==255: 
            image_first_band[i, j]= s_p[i, j]
image_final = cv2.merge((image_first_band, image_first_band, image_first_band))
cv2.imshow('salt_peeper_noise', image_final)
#Maintain output window until user presses a key
cv2.waitKey(0)
cv2.destroyAllWindows()
#%%
#LOW PASS FILTER 
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

p= 0
for q in range(1, new_coloumn-1):
    padded_matrix[p, q]= padded_matrix[p+1, q ]

u= new_row-1
for v in range(1, new_coloumn-1):
    padded_matrix[u, v]= padded_matrix[u-1, v ]

padded_matrix[0, 0]= padded_matrix[0, 1]
padded_matrix[new_row-1, 0]= padded_matrix[new_row-1, 1]
padded_matrix[0, new_row-1] = padded_matrix[0, new_row-2]
padded_matrix[new_row-1, new_row-1]= padded_matrix[new_row-1, new_row-2]

b= 0
for a in range(1, new_coloumn-1 ): 
    padded_matrix[a, b]= padded_matrix[a, b+1]

d= new_coloumn-1
for c in range(1, new_coloumn-1 ): 
    padded_matrix[c, d]= padded_matrix[c, d-1]

LPF_matrix= np.zeros((int(rows), int(coloumns)))

for i in range(0, rows): 
    for j in range(0, coloumns):
        LPF_matrix[i, j]= math.floor((padded_matrix[i, j]*1+padded_matrix[i, j+1]*1+padded_matrix[i, j+2]*1+
                          padded_matrix[i+1, j]*1+padded_matrix[i+1, j+1]*1+padded_matrix[i+1, j+2]*1+
                          padded_matrix[i+2, j]*1+padded_matrix[i+2, j+1]*1+padded_matrix[i+2, j+2]*1)/9)
                          


LPFFiltered= np.uint8(LPF_matrix)
image_final1 = cv2.merge((LPFFiltered, LPFFiltered, LPFFiltered))
cv2.imshow('LPF_Filtered_Image', image_final1)
#Maintain output window until user presses a key
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
#Median FILTER 
import statistics
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

p= 0
for q in range(1, new_coloumn-1):
    padded_matrix[p, q]= padded_matrix[p+1, q ]

u= new_row-1
for v in range(1, new_coloumn-1):
    padded_matrix[u, v]= padded_matrix[u-1, v ]

padded_matrix[0, 0]= padded_matrix[0, 1]
padded_matrix[new_row-1, 0]= padded_matrix[new_row-1, 1]
padded_matrix[0, new_row-1] = padded_matrix[0, new_row-2]
padded_matrix[new_row-1, new_row-1]= padded_matrix[new_row-1, new_row-2]

b= 0
for a in range(1, new_coloumn-1 ): 
    padded_matrix[a, b]= padded_matrix[a, b+1]

d= new_coloumn-1
for c in range(1, new_coloumn-1 ): 
    padded_matrix[c, d]= padded_matrix[c, d-1]

Median_matrix= np.zeros((int(rows), int(coloumns)))

for i in range(0, rows): 
    for j in range(0, coloumns):
        tupleA = (padded_matrix[i, j], padded_matrix[i, j+1], padded_matrix[i, j+2],padded_matrix[i+1, j], padded_matrix[i+1, j+1], padded_matrix[i+1, j+2], padded_matrix[i+2, j], padded_matrix[i+2, j+1], padded_matrix[i+2, j+2])
        Median_matrix[i, j]= statistics.median(tupleA)
                          


MedianFiltered= np.uint8(Median_matrix)
image_final2 = cv2.merge((MedianFiltered, MedianFiltered, MedianFiltered))
cv2.imshow('Median_Filtered_Image', image_final2)
#Maintain output window until user presses a key
cv2.waitKey(0)
cv2.destroyAllWindows()


# %%
