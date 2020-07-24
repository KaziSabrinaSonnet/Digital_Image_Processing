
#%%
#importing cv2
import cv2
# Using cv2.imread() method
img1 = cv2.imread('3_3.jpg')
# Displaying the image using cv2.imshow()
cv2.imshow('3_3.jpg', img1)
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
value_band = hsv_image1[:, :, 2]
rows = value_band .shape[0]
coloumns= value_band.shape[1] 
values = []
for i in range(0, rows): 
    for j in range(0, coloumns):
        values.append(value_band [i, j])
frequencies = {x:values.count(x) for x in values}
import collections
od = collections.OrderedDict(sorted(frequencies.items()))
import numpy as np
probability = []
for item in list(od.values()):
    probability.append(item/sum(list(od.values())))
cp = np.cumsum(probability).tolist()
od_list = list(od.keys())
res = {od_list[i]: cp[i] for i in range(len(od_list))} 

# %%
import math
r = value_band .shape[0]
c = value_band.shape[1]
new_band= np.zeros((int(r), int(c)))
for i in range (0, r): 
    for j in range (0, c): 
        new_band[i, j]= math.floor(res[(value_band [i, j])]*255)
v1 = cv2.normalize(src=new_band, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#%%
h1= hsv_image1[:, :, 0]
s1= hsv_image1[:, :, 1]
new_hsv= cv2.merge((h1,s1,v1))
final_rgb = cv2.cvtColor(new_hsv, cv2.COLOR_HSV2BGR)
# Displaying the image using cv2.imshow()
cv2.imshow('Histogram_Equalization_Image1', final_rgb)
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
plt.title("Histrogram of_histogram_equalized__image02")
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
plt.title("Cumulative Distribution Function of histogram_equalized__image02")
plt.plot(od_list, cp, c='blue')
plt.show()

# %%
