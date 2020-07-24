#%% 
#Problem 2
#Linear Streatching 
#importing cv2
import cv2
# Using cv2.imread() method
img1 = cv2.imread('3_2.jpg')
# Displaying the image using cv2.imshow()
cv2.imshow('3_2.jpg', img1)
#Maintain output window until user presses a key
cv2.waitKey(0)
cv2.destroyAllWindows()
#%%
hsv_image1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
# Displaying the image using cv2.imshow()
cv2.imshow('HSV Image', hsv_image1)
#Maintain output window until user presses a key
cv2.waitKey(0)
cv2.destroyAllWindows()
#%%
import numpy as np 
value1 = hsv_image1[:, :, 2]
minimum = np.amin(value1)
maximum = np.amax(value1)
#the minimum range is 1 and maximum range is 157
r = value1.shape[0]
c = value1.shape[1]
new_value1= np.zeros((int(r), int(c)))
# %%
#Choosing Appropriate value for x1, x2, y1, y2
r1 = 60
r2 = 120 
s1 = 40
s2 = 70
slope1 = s1/r1
slope2 = (s2-s1)/(r2-r1)
slope3= (255-s2)/(255-r2)
"""
intersect1 = y1-slope2*x1
intersect2 = y2-slope3*x2
"""
for i in range(0, r): 
    for j in range (0, c): 
        if 0<=value1[i, j]<=r1: 
            new_value1[i, j]=(slope1*value1[i, j])
        elif r1<=value1[i, j]<=r2:
            new_value1[i, j]=(slope2*(value1[i, j]-r1))+s1
        elif r2<value1[i, j]<255: 
            new_value1[i, j]=(slope2*(value1[i, j]-r2))+s2
v1 = cv2.normalize(src=new_value1, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#%%
h1= hsv_image1[:, :, 0]
s1= hsv_image1[:, :, 1]
new_hsv= cv2.merge((h1,s1,v1))
final_rgb = cv2.cvtColor(new_hsv, cv2.COLOR_HSV2BGR)
# Displaying the image using cv2.imshow()
cv2.imshow('Linearly_Streatched_Image1', final_rgb)
#Maintain output window until user presses a key
cv2.waitKey(0)
cv2.destroyAllWindows()
#%%
#Collecting values from 2D array 
image_first_band = final_rgb[:,:,2]
rows = image_first_band.shape[0]
coloumns= image_first_band.shape[1] 
values = []
for i in range(0, rows): 
    for j in range(0, coloumns):
        values.append(image_first_band[i, j])
frequencies = {x:values.count(x) for x in values}
import collections
od = collections.OrderedDict(sorted(frequencies.items()))
#%%
#Histogram
import matplotlib.pyplot as plt
ax = plt.subplot(111)
w = 0.3
ax.bar(list(od.keys()), list(od.values()) , width=w, color='b', align='center')
ax.autoscale(tight=True)
plt.title("Histrogram of linearly_streatched_image01")
plt.ylim([0, 1200])
plt.xlabel("Pixel Intensity")
plt.ylabel("Pixel Frequency")
plt.show()
#%%
#CDF
import numpy as np
probability = []
for item in list(od.values()):
    probability.append(item/sum(list(od.values())))
cp = np.cumsum(probability).tolist()
od_list = list(od.keys())
amin, amax = min(od_list), max(od_list)
for i, val in enumerate(od_list):
    od_list[i] = (val-amin) / (amax-amin)
plt.xlabel("Pixel Intensity")
plt.ylabel("Cumulative Probability")
plt.title("Cumulative Distribution Function of linearly_streatched_image01")
plt.plot(od_list, cp, c='blue')
plt.show()

# %%
