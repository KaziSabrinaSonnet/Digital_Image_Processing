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
#importing cv2
import cv2
# Using cv2.imread() method
img1 = cv2.imread('target1.jpg')
# Displaying the image using cv2.imshow()
cv2.imshow('target1.jpg', img1)
#Maintain output window until user presses a key
cv2.waitKey(0)
cv2.destroyAllWindows()
#%%
hsv_image2 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
# Displaying the image using cv2.imshow()
cv2.imshow('HSV Image', hsv_image2)
#Maintain output window until user presses a key
cv2.waitKey(0)
cv2.destroyAllWindows()
# %%
#Creating normally distributed transformation function
#or transformation function from target image
import numpy as np
original_value_band = hsv_image1[:, :, 2]
#target_value_band= np.random.binomial(n=255, p=0.5, size=(228, 300))
target_value_band1 = hsv_image2[:, :, 2]
#%%
#Algorithm
import collections
def takeClosest(num,collection):
   return min(collection,key=lambda x:abs(x-num))
intensity1 = []
intensity2 = []
probability1 =[]
probability2 = []
values1 = []
values2 = []
intensity= []
final_list =[]
rows1 = original_value_band .shape[0]
coloumns1= original_value_band.shape[1] 
for i in range(0, rows1): 
    for j in range(0, coloumns1):
        values1.append(original_value_band[i, j])
frequencies1 = {x1:values1.count(x1) for x1 in values1}

rows2 = target_value_band1.shape[0]
coloumns2= target_value_band1.shape[1] 
for i in range(0, rows2): 
    for j in range(0, coloumns2):
        values2.append(target_value_band1[i, j])
frequencies2 = {x2:values2.count(x2) for x2 in values2}
for item in range(0, 256):
    if item in frequencies1: 
        intensity1.append(frequencies1[item])
    else: 
        intensity1.append(0)
for item in range(0, 256):
    intensity.append(item)
    if item in frequencies2: 
        intensity2.append(frequencies2[item])
    else: 
        intensity2.append(0)
for item in intensity1:
    probability1.append(item/sum(intensity1))
for item in intensity2:
    probability2.append(item/sum(intensity2))

cp1 = np.cumsum(probability1).tolist()
cp2 = np.cumsum(probability2).tolist()


res = {cp2[i]: intensity[i] for i in range(len(cp2))} 

for item in cp1:
    final_list.append(res[takeClosest(item,cp2)])

#intensity is mapped to final list


# %%
resf= {intensity[i]: final_list[i] for i in range(len(intensity))}
new_original_band= np.zeros((int(rows1), int(coloumns1)))
for i in range(0, rows1):
    for j in range(0,coloumns1 ): 
        new_original_band[i, j] = resf[(original_value_band[i, j])]
v1 = cv2.normalize(src=new_original_band, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#%%
h1= hsv_image1[:, :, 0]
s1= hsv_image1[:, :, 1]
new_hsv= cv2.merge((h1,s1,v1))
final_rgb = cv2.cvtColor(new_hsv, cv2.COLOR_HSV2BGR)
# Displaying the image using cv2.imshow()
cv2.imshow('Histogram_Specification_from_target_image', final_rgb)
#Maintain output window until user presses a key
cv2.waitKey(0)
cv2.destroyAllWindows()
# %%
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
plt.title("Histrogram of_histogram_specified_image01")
plt.xlabel("Pixel Intensity")
plt.ylim(0, 7000)
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
plt.title("Cumulative Distribution Function of histogram_specified_image01")
plt.plot(od_list, cp, c='blue')
plt.show()


# %%
