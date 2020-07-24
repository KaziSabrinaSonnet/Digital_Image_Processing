#%%
#Problem: 02
#importing cv2
import cv2
# Using cv2.imread() method
img = cv2.imread('4_2.bmp')
# Displaying the image using cv2.imshow()
cv2.imshow('Original Image', img)
#Maintain output window until user presses a key
cv2.waitKey(0)
cv2.destroyAllWindows()
# %%
img1= img[:, :, 0]
img2= img[:, :, 1]
img3= img[:, :, 2]
# %%
row = img1.shape[0]
coloumn= img1.shape[1]
# %%
#First_Band
#Extracting a 3 by 3 patch around each pixel and forming matrix
List1 = []
for i in range(1, row-1):
    for j in range(1, coloumn-1):
        list1=[]
        list1.append(img1[i-1, j-1])
        list1.append(img1[i-1, j]) 
        list1.append(img1[i-1, j+1])
        list1.append(img1[i, j-1])
        list1.append(img1[i, j])
        list1.append(img1[i, j+1])
        list1.append(img1[i+1, j-1])
        list1.append(img1[i+1, j])
        list1.append(img1[i+1, j+1])
        List1.append(list1)
matrix1 = np.array(List1)
mask1= np.array([[-1], [-1], [-1], [-1],[9],[-1], [-1], [-1], [-1]]) #HPF MASK
C1=np.matmul(matrix1, mask1) #Convolition mask and matrix
C1A = C1.reshape(row-2, coloumn-2) #reshape to (height-2, width-2)
C1A= np.uint8(C1A)
# %%
#Second_Band
#Extracting a 3 by 3 patch around each pixel and forming matrix
List2 = []
for i in range(1, row-1):
    for j in range(1, coloumn-1):
        list2=[]
        list2.append(img2[i-1, j-1])
        list2.append(img2[i-1, j]) 
        list2.append(img2[i-1, j+1])
        list2.append(img2[i, j-1])
        list2.append(img2[i, j])
        list2.append(img2[i, j+1])
        list2.append(img2[i+1, j-1])
        list2.append(img2[i+1, j])
        list2.append(img2[i+1, j+1])
        List2.append(list2)
matrix2 = np.array(List2)
mask2= np.array([[-1], [-1], [-1], [-1],[9],[-1], [-1], [-1], [-1]]) #HPF MASK
C2=np.matmul(matrix2, mask2) #Convolition mask and matrix
C1B = C2.reshape(row-2, coloumn-2) #reshape to (height-2, width-2)
C1C= np.uint8(C1B)
# %%
#Third_Band
#Extracting a 3 by 3 patch around each pixel and forming matrix
List3 = []
for i in range(1, row-1):
    for j in range(1, coloumn-1):
        list3=[]
        list3.append(img3[i-1, j-1])
        list3.append(img3[i-1, j]) 
        list3.append(img3[i-1, j+1])
        list3.append(img3[i, j-1])
        list3.append(img3[i, j])
        list3.append(img3[i, j+1])
        list3.append(img3[i+1, j-1])
        list3.append(img3[i+1, j])
        list3.append(img3[i+1, j+1])
        List3.append(list3)
matrix3 = np.array(List3)
mask3= np.array([[-1], [-1], [-1], [-1],[9],[-1], [-1], [-1], [-1]]) #HPF MASK
C3=np.matmul(matrix3, mask3) #Convolition mask and matrix
C1D = C3.reshape(row-2, coloumn-2) #reshape to (height-2, width-2)
C1E= np.uint8(C1D)

# %%
image_final3 = cv2.merge((C1A, C1C, C1E))
cv2.imshow('Problem2', image_final3)
#Maintain output window until user presses a key
cv2.waitKey(0)
cv2.destroyAllWindows()



# %%
