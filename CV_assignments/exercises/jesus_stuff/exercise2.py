# # # # # # # # # # # # # # # # # # 
#								  #
# AUTHOR: Jesús García Fernández  #
# ID: 6230556					  #
#                                 #
# # # # # # # # # # # # # # # # # # 

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat



########################## FUNCTIONS ##########################


def manual_convolve_img(img, kernel=[]):
	#Converting to gray scale
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#Normalizing img	
	img = img/255
	#Selecting kernel
	if len(kernel)==0:
		#By default borders detection
		kernel = np.array([[1,0,-1],  
						  [1,0,-1],
						  [1,0,-1]])
	#Adding zero padding
	for _ in range(int((len(kernel)-1)/2)):
		img = np.column_stack((img, [0]*len(img))) 
		img = np.column_stack(([0]*len(img), img)) 
		img = np.ma.row_stack((img, [0]*len(img[0]))) 
		img = np.ma.row_stack(([0]*len(img[0]), img)) 
	#Convolution stuff
	conv_img = img.copy()
	for idx_row in range(len(img)-(len(kernel)-1)):
		for idx_column in range(len(img[0])-(len(kernel)-1)):
			#Taking crop size kernel
			cut_img = img[idx_row:idx_row+len(kernel),idx_column:idx_column+len(kernel)]
			#Convolving
			summ = np.sum(cut_img*kernel)
			conv_img[idx_row,idx_column]=summ
	return conv_img, img




########################## (a) ##########################


#Reading file
path = 'exercise2_edges/Lena.jpg'
img = cv2.imread(path)



#### Kernels
v_kernel = np.array([[-1,0,1],  
					  [-1,0,1],
					  [-1,0,1]])

h_kernel = np.array([[-1,-1,-1],  
					  [0,0,0],
					  [1,1,1]])



##### CONVOLVING HORIZONTAL
print("Horizantal filter convolution...")
# plt.subplot(2,1,1)
img_conv_h,img_gray = manual_convolve_img(img, h_kernel)
# plt.imshow(img_gray, cmap='gray')
# plt.title("Original")

# plt.subplot(2,1,2)
# plt.imshow(img_conv_h, cmap='gray')
# plt.title("Convoluted manually")
# plt.tight_layout()
# plt.show()


##### CONVOLVING VERTICAL
print("Vertical filter convolution...")
# plt.subplot(2,1,1)
img_conv_v,img_gray = manual_convolve_img(img, v_kernel)
# plt.imshow(img_gray, cmap='gray')
# plt.title("Original")

# plt.subplot(2,1,2)
# plt.imshow(img_conv_v, cmap='gray')
# plt.title("Convoluted manually")
# plt.tight_layout()
# plt.show()


#### CALCULATING GRADIENT
print("HCalculating gradient...")
img_grad =  np.sqrt((img_conv_v**2+img_conv_h**2))
# plt.subplot(2,1,1)
# plt.imshow(img_gray, cmap='gray')
# plt.title("Original")

# plt.subplot(2,1,2)
# plt.imshow(img_grad, cmap='gray')
# plt.title("GRADIENT")
# plt.tight_layout()
# plt.show()


#### BINARIZING
print("Binarizing image with different thresholds...")
#Taking random thresholds
thresh = [0.05, 0.25, 0.5, 0.75, 1]

#Showing pictures
plt.figure(figsize=(10, 8))
plt.subplot(3,3,1)
plt.title('Original')
plt.imshow(img_gray, cmap='gray')
plt.subplot(3,3,2)
plt.title('After applying gradient')
plt.imshow(img_grad, cmap='gray')
for i in range(5):
	_ , img_bw = cv2.threshold(img_grad, thresh[i], 255, cv2.THRESH_BINARY)
	plt.subplot(3,3,i+3)
	plt.imshow(img_bw, cmap='gray')
	plt.title("Threshold:"+str(thresh[i]))	
plt.tight_layout()
plt.savefig('Result_exercise2a.png', dpi = 300)
plt.show()




########################## (c) ##########################

print("Applying Laplacing filters...")

#Laplacing filters
kernel_3 = np.array([[0,1,0],[1,-4,1],[0,1,0]])
kernel_5 = loadmat('exercise2_edges/Log5.mat')['Log5']
kernel_17 = loadmat('exercise2_edges/Log17.mat')['Log17']

#Showing results
plt.figure(figsize=(10, 8))

#Filter 3x3
img_conv, img_gray = manual_convolve_img(img, kernel_3)
plt.subplot(2,2,1)
plt.title("3x3 filter")
plt.imshow(img_conv, cmap='gray')
#Filter 5x5
img_conv, img_gray = manual_convolve_img(img, kernel_5)
plt.subplot(2,2,2)
plt.title("5x5 filter")
plt.imshow(img_conv, cmap='gray')
#Filter 17x17
img_conv, img_gray = manual_convolve_img(img, kernel_17)
plt.subplot(2,2,3)
plt.title("17x17 filter")
plt.imshow(img_conv, cmap='gray')

plt.tight_layout()
plt.savefig('Result_exercise2c.png', dpi = 300)
plt.show()




