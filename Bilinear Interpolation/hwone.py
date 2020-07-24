#%%
#importing cv2
import cv2
# Using cv2.imread() method
img = cv2.imread('1_4.bmp')
# Displaying the image using cv2.imshow()
cv2.imshow('Lena_Image', img)
#Maintain output window until user presses a key
cv2.waitKey(0)
cv2.destroyAllWindows()
#%%
#savingimage
isWritten = cv2.imwrite('LenaImage.png', img) 
#%%
#getting the maximum and minimum value for the image
import numpy as np 
#getting the type of loaded image data
print('Lena_Image dtype ', img.dtype)
minimum = np.amin(img)
maximum = np.amax(img)
#%%
#converting data to double type 
print(img.astype(np.float))
#%%
cv2.imshow('Lena_Image', img.astype(np.float))
cv2.waitKey(0)
cv2.destroyAllWindows()
#%%
isWritten = cv2.imwrite('DoubleTypeLenaImage.png', img.astype(np.float)) 
#%%
import cv2
#show double type image 
#convert to normalized floating point
out = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
cv2.imshow('Double_Type_Lena_Image', out) 
cv2.waitKey(0)
cv2.destroyAllWindows()
#%%
isWritten = cv2.imwrite('NormalizedLenaImage.png', out) 
#%%
X= cv2.imread('1_2.tif')
Y = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray Image', Y)
cv2.waitKey(0)
cv2.destroyAllWindows()
#%%
isWritten = cv2.imwrite('GrayFlowerImage.png', Y) 
#%%
#find out height and width of image
(height, width) = Y.shape[:2]
# calculate center of image
center = (height/2, width/2)
scale= 1.0
angle = -120 #clockwise (negative)
M = cv2.getRotationMatrix2D(center, angle, scale)
Z0 = cv2.warpAffine(Y, M, (height, width))
cv2.imshow('Rotated Image', Z0)
cv2.waitKey(0) 
cv2.destroyAllWindows()
#%%
isWritten = cv2.imwrite('120DRotatedImage.png', Z0) 
#%%
#Rotate 12 times
i= 1
while i<13:
    (height, width) = Y.shape[:2]
    center = (height/2, width/2)
    scale= 1.0
    angle = -10
    M = cv2.getRotationMatrix2D(center, angle, scale)
    Y = cv2.warpAffine(Y, M, (height, width))
    i= i+1
Z1 = Y
cv2.imshow('Rotated_Image_12_times', Z1)
cv2.waitKey(0) 
cv2.destroyAllWindows()
#%%
isWritten = cv2.imwrite('10T12DRotatedImage.png', Z1) 
#%%
import cv2
import numpy as np 
X=np.loadtxt('1_3.asc')
out0 = cv2.normalize(X, None, 0.0, 1.0, cv2.NORM_MINMAX)
cv2.imshow('1_3', out0) 
cv2.waitKey(0)
cv2.destroyAllWindows()
#%%
isWritten = cv2.imwrite('1_3.png', out0) 
#%%
#reducig image size image by throw away technique
X.shape
Z = X[0:384:4,0:256:4]
Z.shape
#%%
Y1 = cv2.normalize(Z, None, 0.0, 1.0, cv2.NORM_MINMAX)
cv2.imshow('Reduces_Image_1', Y1) 
cv2.waitKey(0)
cv2.destroyAllWindows()
#%%
isWritten = cv2.imwrite('PixelReduction1.png', Y1) 
#%%
#reducig image size image by average technique
import numpy as np
R=4 #given
rows = X.shape[0]
coloumns = X.shape[1]
s= (int(rows/R), int(coloumns/R))
z= np.zeros(s)
rows_z = z.shape[0]
coloumns_z = z.shape[1]
i= 0
j=0
for m in range(0, rows_z): 
    for n in range(0, coloumns_z):
        if j<256:
            z[m, n] = (X[i, j]+X[i+1, j]+X[i+2, j]+X[i+3, j]+X[i, j+1]+X[i+1, j+1]+X[i+2, j+1]+X[i+3, j+1]+X[i, j+2]+
                       X[i+1, j+2]+ X[i+2, j+2]+ X[i+3, j+2]+ X[i, j+3]+X[i+1, j+3]+X[i+2, j+3]+X[i+3, j+3])
            z[m, n] = z[m, n]/16
            j=j+R
        elif i<384:
            i=i+R
            j=0
            z[m, n] = (X[i, j]+X[i+1, j]+X[i+2, j]+X[i+3, j]+X[i, j+1]+X[i+1, j+1]+X[i+2, j+1]+X[i+3, j+1]+
                       X[i, j+2]+X[i+1, j+2]+ X[i+2, j+2]+ X[i+3, j+2]+ X[i, j+3]+X[i+1, j+3]+X[i+2, j+3]+X[i+3, j+3])
            z[m, n] = z[m, n]/16
            j=j+R
Y2 = cv2.normalize(z, None, 0.0, 1.0, cv2.NORM_MINMAX)
cv2.imshow('Reduces_Image_2', Y2) 
cv2.waitKey(0)
cv2.destroyAllWindows()
#%%
isWritten = cv2.imwrite('PixelAverageTechnique.png', Y2) 
#%%
#enlarging image size by repeating technique 
import numpy as np
R=4 #given
r = Z.shape[0]
c = Z.shape[1]
t= (int(r*R), int(c*R))
z1= np.zeros(t)
rows_z1 = z1.shape[0]
coloumns_z1 = z1.shape[1]
i= 0
j=0
for m in range(0, r): 
    for n in range(0, c):
        if j<256:
            z1[i, j] = Z[m,n]
            z1[i+1, j] = Z[m,n]
            z1[i+2, j] = Z[m,n]
            z1[i+3, j] = Z[m,n]
            z1[i, j+1] = Z[m,n]
            z1[i+1, j+1] = Z[m,n]
            z1[i+2, j+1] = Z[m,n]
            z1[i+3, j+1] = Z[m,n]
            z1[i, j+2]= Z[m,n]
            z1[i+1, j+2]=Z[m,n] 
            z1[i+2, j+2]= Z[m,n]
            z1[i+3, j+2]= Z[m,n]
            z1[i, j+3]= Z[m,n]
            z1[i+1, j+3]=Z[m,n] 
            z1[i+2, j+3]= Z[m,n]
            z1[i+3, j+3]= Z[m,n]
            j= j+R
        elif i<384:
            i=i+R
            j=0
            z1[i, j] =Z[m,n]
            z1[i+1, j] = Z[m,n]
            z1[i+2, j] = Z[m,n]
            z1[i+3, j] = Z[m,n]
            z1[i, j+1] = Z[m,n]
            z1[i+1, j+1] = Z[m,n]
            z1[i+2, j+1] = Z[m,n] 
            z1[i+3, j+1] = Z[m,n]
            z1[i, j+2]= Z[m,n]
            z1[i+1, j+2]= Z[m,n]
            z1[i+2, j+2]= Z[m,n]
            z1[i+3, j+2]= Z[m,n]
            z1[i, j+3]=Z[m,n] 
            z1[i+1, j+3]= Z[m,n]
            z1[i+2, j+3]=Z[m,n] 
            z1[i+3, j+3]= Z[m,n]
            j= j+R
E1 = cv2.normalize(z1, None, 0.0, 1.0, cv2.NORM_MINMAX)
cv2.imshow('Enlarged_Image_1', E1) 
cv2.waitKey(0)
cv2.destroyAllWindows()
#%%
isWritten = cv2.imwrite('PixelRepeating.png', E1) 
#%%
#Bilinear Interpolation 
from math import floor
def round_down(num,mul):
    if num%mul==0: 
        res = num-mul
    else:
        res = floor(num / mul)* mul
    return res 
R= 4
rows_Z = Z.shape[0]
coloumns_Z = Z.shape[1]
o= (int(rows_Z*R), int(coloumns_Z*R))
new_Z= np.zeros(o)
rows_newZ = new_Z.shape[0]
coloumns_newZ = new_Z.shape[1]
i= 0
j=0
#making 4 point boundaries 
for m in range(0,rows_Z): 
    for n in range(0, coloumns_Z): 
        if j<256: 
            new_Z[i, j] = Z[m, n]
            j = j+4
        elif i<384:
            i = i+R
            j=0
            new_Z[i, j] = Z[m, n]
            j= j+R       
#calculating points from boundaries 
for r in range(0, (((rows_Z-1)*R)+1)): 
    for s in range(0,(((coloumns_Z-1)*R)+1) ): 
        if new_Z[r, s]== 0:
            ax= round_down(r, R)
            ay= round_down(s, R)
            bx= ax
            by= ay+R
            cx= bx+R
            cy= by
            dx= cx
            dy= cy-R
            minx= min(ax, bx, cx, dx)
            maxx= max(ax, bx, cx, dx)
            miny= min(ay, by, cy, dy)
            maxy= max(ay, by, cy, dy)
            originalx= r
            originaly= s
            normalizedx=(originalx-minx)/(maxx-minx) 
            normalizedy=(originaly-miny)/(maxy-miny) 
            new_Z[r, s] = (new_Z[bx, by]-new_Z[ax, ay])*normalizedx + (new_Z[dx, dy]-new_Z[ax, ay])*normalizedy + ((new_Z[ax, ay]-new_Z[bx, by]-new_Z[dx, dy]+new_Z[ax, ay])*normalizedx*normalizedy)+ new_Z[ax, ay]

#taking care of points outside grid 
for u in range (0, rows_newZ): 
    for v in range ((((coloumns_Z-1)*R)+1), coloumns_newZ): 
        new_Z[u, v]=new_Z[u, ((coloumns_Z-1)*R)]

for p in range ((((rows_Z-1)*R)+1), rows_newZ):
    for q in range (0, coloumns_newZ):
        new_Z[p, q]=new_Z[((rows_Z-1)*R), q]

#%%
E2 = cv2.normalize(new_Z, None, 0.0, 1.0, cv2.NORM_MINMAX)
cv2.imshow('Enlarged_Image_2', E2) 
cv2.waitKey(0)
cv2.destroyAllWindows()
# %%
isWritten = cv2.imwrite('Bilinear Interpolation.png', E2) 


# %%
