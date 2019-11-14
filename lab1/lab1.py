import numpy as np 
import cv2
from matplotlib import pyplot as plt

def contrast_strect(imagename):
	imgg = cv2.imread(str(imagename),0)
	a = 0
	b = 255
	c = np.min(imgg)
	d = np.max(imgg)
	minmax_img = np.zeros((imgg.shape[0],imgg.shape[1]),dtype = 'uint8')
	for i in range(imgg.shape[0]):
	    for j in range(imgg.shape[1]):
	        minmax_img[i,j] = (b-a)*(imgg[i,j]-c)/(d-c)
	return minmax_img
minmax_img = contrast_strect('SanFrancisco.jpg')
img = cv2.imread('SanFrancisco.jpg',0)
target = [img,minmax_img]
for i in range(len(target)):
	plt.figure()
	plt.subplot(2,1,1)
	plt.imshow(target[i],'gray')
	plt.suptitle('aaaaa')
	plt.subplot(2,1,2)
	plt.hist(target[i].ravel(),256,[0,256])
	plt.xticks([]),plt.yticks([])
plt.show()
cv2.imshow('original',img)
cv2.waitKey(0)
cv2.imshow('Minmax',minmax_img)
cv2.waitKey(0)
cv2.destroyAllWindows()



# def sobel(imgarr,kernelx,Kernely):
# 	img = cv2.imread(str(imgarr),0)
# 	blur1 = cv2.filter2D(img,-1,Kernelx)
# 	blur2 = cv2.filter2D(img,-1,Kernely)
# 	plt.figure()
# 	plt.subplot(1,2,1)
# 	plt.imshow(blur1,'gray')
# 	plt.title('X direction')
# 	plt.subplot(1,2,2)
# 	plt.imshow(blur2,'gray')
# 	plt.title('Y direction')
# 	plt.xticks([]),plt.yticks([])
# 	direct = ['x','y']
# 	tar = [blur1,blur2]
# 	for i in range(len(tar)):
# 		cv2.imwrite('{}_direction.png'.format(direct[i]),tar[i])

# Kernelx = np.array([[-1,0,1],[ -2,0,2],[ -1,0,1]])

# Kernely = np.array([[-1,-2,-1],[ 0,0,0],[ 1,2,1]])
# imgarr = 'SanFrancisco.jpg'
# img = cv2.imread(str(imgarr),0)
# image_x = cv2.Sobel(img,cv2.CV_8U,1,0)
# image_y = cv2.Sobel(img,cv2.CV_8U,0,1)
# cv2.imshow('X',image_x)
# cv2.waitKey(0)
# cv2.imshow('Y',image_y)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# sh = sobel('SanFrancisco.jpg',Kernelx,Kernely)
# plt.show()

