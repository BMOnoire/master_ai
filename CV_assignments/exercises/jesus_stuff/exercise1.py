# # # # # # # # # # # # # # # # # # 
#								  #
# AUTHOR: Jesús García Fernández  #
# ID: 6230556					  #
#                                 #
# # # # # # # # # # # # # # # # # # 

import cv2
import numpy as np
import matplotlib.pyplot as plt


########################## (b) ##########################

#Generating random numbers
y = []
for _ in range(100): y.append(np.random.random()*2)

#Creating matrix
A=[y]*100

#Showing picture
plt.imshow(A)
plt.title("Random picture")
plt.show()


########################## (c) ##########################

#Reading file
path = 'exercise1_basics/lab1a.png'
img = cv2.imread(path)

#Showing image
#plt.imshow(img[:,:,-1])
plt.title("Reading picture")
plt.show()


########################## (d) ##########################

#Showing image gray
img_gray = cv2.imread(path,0)
plt.imshow(img_gray, cmap='gray')
plt.title("Gray picture")
plt.show()


########################## (e) ##########################

#Reading and transforming image to HSV
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#Showing pictures
plt.subplot(2,2,1)
plt.title('Original')
plt.imshow(img[:,:,-1])
plt.subplot(2,2,2)
plt.title('First channel')
plt.imshow(hsv_img[:,:,0])
plt.subplot(2,2,3)
plt.title('Second channel')
plt.imshow(hsv_img[:,:,1])
plt.subplot(2,2,4)
plt.title('Third channel')
plt.imshow(hsv_img[:,:,2])
plt.show()


########################## (f) ##########################

#Taking random thresholds
thresh = [np.random.randint(1,255) for _ in range(5)]

#Showing pictures
plt.subplot(2,3,1)
plt.title('Original')
plt.imshow(img[:,:,::-1])
for i in range(5):
	_ , img_bw = cv2.threshold(img_gray, thresh[i], 255, cv2.THRESH_BINARY)
	plt.subplot(2,3,i+2)
	plt.imshow(img_bw, cmap='gray')
	plt.title("Threshold:"+str(thresh[i]))	
plt.show()


########################## (g) ##########################

# Taking a rectangle
x=250
y=320

plt.figure(figsize=(10, 8))

#original size
cv2.rectangle(img,(x,y),(x+150,y+100),(0,255,0),2)
plt.subplot(3,2,1)
plt.title("Original size")
plt.imshow(img[:,:,::-1])

#Scalating
img_2 = cv2.resize(img, (0,0), fx=1, fy=0.5)
plt.subplot(3,2,2)
plt.title("Resized 1")
plt.imshow(img_2[:,:,::-1])

#Scalating
img_2 = cv2.resize(img, (0,0), fx=0.5, fy=1)
plt.subplot(3,2,3)
plt.title("Resized 2")
plt.imshow(img_2[:,:,::-1])


#Magnifying bounding box
plt.subplot(3,2,4)
plt.title("Bounding box")
plt.imshow(img[y:y+100,x:x+150,::-1])


#Magnifying and resized bounding box

img_3 = cv2.resize(img[y:y+100,x:x+150,::-1], (0,0), fx=0.5, fy=1)
plt.subplot(3,2,5)
plt.title("Bounding box resized")
plt.imshow(img_3)

plt.tight_layout()
plt.show()



