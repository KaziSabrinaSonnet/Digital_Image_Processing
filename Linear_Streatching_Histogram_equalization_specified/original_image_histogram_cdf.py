#%%
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
#importing cv2
import cv2
# Using cv2.imread() method
img2 = cv2.imread('3_3.jpg')
# Displaying the image using cv2.imshow()
cv2.imshow('3_3.jpg', img2)
#Maintain output window until user presses a key
cv2.waitKey(0)
cv2.destroyAllWindows()
#%%
#Collecting values from 2D array 
image_first_band = img2[:,:,0]
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
plt.title("Histrogram of original_image02")
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
plt.title("Cumulative Distribution Function of original image02")
plt.plot(od_list, cp, c='blue')
plt.show()


# %%
