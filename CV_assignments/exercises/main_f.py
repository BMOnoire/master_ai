import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = ".\\SecondWeekExercises\\exercise1_basics\\lab1a.png"
img = cv2.imread(image_path)   # reads an image in the BGR format
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)   # BGR -> HSV

plt.subplot(2, 2, 1), plt.imshow(img, cmap="hsv"), plt.title('Original')

img_h = img[ :, :, 0 ]
img_s = img[ :, :, 1 ]
img_v = img[ :, :, 2 ]

plt.subplot(2, 2, 2), plt.imshow(img_h, cmap="hsv"), plt.title('H')
plt.subplot(2, 2, 3), plt.imshow(img_s, cmap="hsv"), plt.title('S')
plt.subplot(2, 2, 4), plt.imshow(img_v, cmap="hsv"), plt.title('V')


#imgplot = plt.imshow(img,)
plt.show()
print(img.shape)
