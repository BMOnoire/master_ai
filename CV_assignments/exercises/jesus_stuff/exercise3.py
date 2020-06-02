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

def manual_convolve_img(img, kernel=[], x=True):
	#Converting to gray scale
	if x:
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



def equalize_histogram(img):
	L=256
	#Converting to gray scale
	img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)	
	#Getting histogram
	img_hist, bins = np.histogram(img_g, bins=256)
	n_pixels = np.prod(img_g.shape)
	#Calculating prob pixels
	prob_pixel = [0]*L 
	summ=0
	for i in range(L):
		summ+=img_hist[i]/n_pixels
		prob_pixel[i]=summ
	#Converting pixels
	new_img = img_g.copy()
	for y in range(len(img_g)):
		for x in range(len(img_g[0])):
			new_img[y,x] = int(((L-1) * prob_pixel[new_img[y,x]])) 
	return new_img



########################## PART 1 (a) ##########################

#Reading file
path = 'exercise3_enhancement-restoration/Unequalized_H.jpg'
img = cv2.imread(path)

#Converting to gray scale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#Getting histogram
img_hist, _ = np.histogram(img_gray, bins=256)

#Equalizing image
new_img = equalize_histogram(img)
#Getting histogram
img_hist_eq, _ = np.histogram(new_img, bins=256)


#Showing comparison original and equalized
#Original image
plt.subplot(2,2,1)
plt.imshow(img_gray, cmap='gray')
plt.title("Original image")
plt.subplot(2,2,2)
plt.hist(img_hist, bins='auto')
plt.title("Histogram original image")
#Equalized image
plt.subplot(2,2,3)
plt.imshow(new_img, cmap='gray')
plt.title("Equalized image")
plt.subplot(2,2,4)
plt.hist(img_hist_eq, bins='auto')
plt.title("Histogram equalized image")
	
plt.tight_layout()
plt.savefig('Result_exercise3a.png', dpi = 300)
plt.show()



########################## PART 1 (b) ##########################


kernel = np.array([[1/9,1/9,1/9],  
				  [1/9,1/9,1/9],
				  [1/9,1/9,1/9]])

kernel_1D = np.array([[0,0,0],  
					  [1/9,1/9,1/9],
					  [0,0,0]])



img_conv,_ = manual_convolve_img(img, kernel)
img_conv2,_ = manual_convolve_img(img, kernel_1D)
img_conv3,_ = manual_convolve_img(img_conv2, np.transpose(kernel_1D), x=False)


#Showing results
plt.subplot(2,2,1)
plt.imshow(img_gray, cmap='gray')
plt.title("Original")
plt.subplot(2,2,2)
plt.imshow(img_conv, cmap='gray')
plt.title("Convolution 2D filter")
#Equalized image
plt.subplot(2,2,3)
plt.imshow(img_conv2, cmap='gray')
plt.title("First convolution 1D filter")
plt.subplot(2,2,4)
plt.imshow(img_conv3, cmap='gray')
plt.title("Second convolution 1D filter")
	
plt.tight_layout()
plt.savefig('Result_exercise3b.png', dpi = 300)
plt.show()



########################## PART 1 (c) ##########################

#Filter high pass
kernel_hp = np.array([[0,-1,0],  
					  [-1,4,-1],
					  [0,-1,2]])

img_conv_hp,_ = manual_convolve_img(img, kernel_hp)

#Showing results
plt.subplot(2,1,1)
plt.imshow(img_gray, cmap='gray')
plt.title("Original")
plt.subplot(2,1,2)
plt.imshow(img_conv_hp, cmap='gray')
plt.title("Convoluted hp")

plt.tight_layout()
plt.savefig('Result_exercise3c.png', dpi = 300)
plt.show()



########################## PART 2 (a) ##########################