import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
def no_filtering_and_thresholding(image_name,vval,mavl,admean_block,Gaussian_block,c):
	img = cv.imread(str(image_name),0)
	titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV','AdaptMean({})'.format(admean_block), 'AdaptGaussian({})'.format(Gaussian_block),"OtsuThres"]
	ret1,th1 = cv.threshold(img,vval,mavl,cv.THRESH_BINARY)
	ret2,th2 = cv.threshold(img,vval,mavl,cv.THRESH_BINARY_INV)
	ret3,th3 = cv.threshold(img,vval,mavl,cv.THRESH_TRUNC)
	ret4,th4 = cv.threshold(img,vval,mavl,cv.THRESH_TOZERO)
	ret5,th5 = cv.threshold(img,vval,mavl,cv.THRESH_TOZERO_INV)
	th6 = cv.adaptiveThreshold(img,mavl,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,admean_block,c)
	th7 = cv.adaptiveThreshold(img,mavl,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,Gaussian_block,c)
	ret6,th8 = cv.threshold(img,0,mavl,cv.THRESH_BINARY+cv.THRESH_OTSU)
	plt.figure()
	plt.suptitle('Three typyes of Thresholding with No filtering')
	IMG = [img,th1,th2,th3,th4,th5,th6,th7,th8]
	for i in range(len(IMG)):
		plt.subplot(3,3,i+1)
		plt.imshow(IMG[i],'gray')
		plt.title(titles[i])
		plt.xticks([]),plt.yticks([])
	plt.figure()
	plt.suptitle('Histogram for thresholding with 25/35 (Mean/Gaussian)')
	himg = [img,th1,th3,th4,th6,th7,th8]
	htitles = ['Original Image','BINARY','TRUNC','TOZERO','AdaptMean({})'.format(admean_block), 'AdaptGaussian({})'.format(Gaussian_block),"OtsuThres"]
	for j in range(len(himg)):
		plt.subplot(3,3,j+1)
		plt.hist(himg[j].ravel(),256,[0,256])
		plt.title(htitles[j])
		plt.xticks([]),plt.yticks([])
def med_filtering_and_thresholding(image_name,vval,mavl,medpara,admean_block,Gaussian_block,c):
	img = cv.imread(str(image_name),0)
	img_blur = cv.medianBlur(img,medpara)
	titles = ['Original Image(blurred)','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV','AdaptMean', 'AdaptGaussian',"OtsuThres"]
	ret1,th1 = cv.threshold(img_blur,vval,mavl,cv.THRESH_BINARY)
	ret2,th2 = cv.threshold(img_blur,vval,mavl,cv.THRESH_BINARY_INV)
	ret3,th3 = cv.threshold(img_blur,vval,mavl,cv.THRESH_TRUNC)
	ret4,th4 = cv.threshold(img_blur,vval,mavl,cv.THRESH_TOZERO)
	ret5,th5 = cv.threshold(img_blur,vval,mavl,cv.THRESH_TOZERO_INV)
	th6 = cv.adaptiveThreshold(img_blur,mavl,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,admean_block,c)
	th7 = cv.adaptiveThreshold(img_blur,mavl,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,Gaussian_block,c)
	ret6,th8 = cv.threshold(img_blur,0,mavl,cv.THRESH_BINARY+cv.THRESH_OTSU)
	plt.figure()
	plt.suptitle('Three typyes of Thresholding under Median filtering({})'.format(medpara))
	IMG = [img_blur,th1,th2,th3,th4,th5,th6,th7,th8]
	for i in range(len(IMG)):
		plt.subplot(3,3,i+1)
		plt.imshow(IMG[i],'gray')
		plt.title(titles[i])
		plt.xticks([]),plt.yticks([])
	plt.figure()
	plt.suptitle('Histogram of Thresholding under Median filtering({})'.format(medpara))
	himg = [img_blur,th1,th3,th4,th6,th7,th8]
	htitles = ['Original Image(blurred)','BINARY','TRUNC','TOZERO','AdaptMean', 'AdaptGaussian',"OtsuThres"]
	for j in range(len(himg)):
		plt.subplot(3,3,j+1)
		plt.hist(himg[j].ravel(),256,[0,256])
		plt.title(htitles[j])
		plt.xticks([]),plt.yticks([])
def Gaussian_filtering_and_thresholding(image_name,vval,mavl,Gaussianparasize,Gaussianparacon,admean_block,Gaussian_block,c):
	img = cv.imread(str(image_name),0)
	img_blur = cv.GaussianBlur(img,(Gaussianparasize,Gaussianparasize),Gaussianparacon)
	titles = ['Original Image(blurred)','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV','AdaptMean', 'AdaptGaussian',"OtsuThres"]
	ret1,th1 = cv.threshold(img_blur,vval,mavl,cv.THRESH_BINARY)
	ret2,th2 = cv.threshold(img_blur,vval,mavl,cv.THRESH_BINARY_INV)
	ret3,th3 = cv.threshold(img_blur,vval,mavl,cv.THRESH_TRUNC)
	ret4,th4 = cv.threshold(img_blur,vval,mavl,cv.THRESH_TOZERO)
	ret5,th5 = cv.threshold(img_blur,vval,mavl,cv.THRESH_TOZERO_INV)
	th6 = cv.adaptiveThreshold(img_blur,mavl,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,admean_block,c)
	th7 = cv.adaptiveThreshold(img_blur,mavl,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,Gaussian_block,c)
	ret6,th8 = cv.threshold(img_blur,0,mavl,cv.THRESH_BINARY+cv.THRESH_OTSU)
	plt.figure()
	plt.suptitle('Three typyes of Thresholding under Gaussian filtering({})'.format(Gaussianparasize))
	IMG = [img_blur,th1,th2,th3,th4,th5,th6,th7,th8]
	for i in range(len(IMG)):
		plt.subplot(3,3,i+1)
		plt.imshow(IMG[i],'gray')
		plt.title(titles[i])
		plt.xticks([]),plt.yticks([])
	plt.figure()
	plt.suptitle('Histogram of Thresholding under Gaussian filtering({})'.format(Gaussianparasize))
	himg = [img_blur,th1,th3,th4,th6,th7,th8]
	htitles = ['Original Image(blurred)','BINARY','TRUNC','TOZERO','AdaptMean', 'AdaptGaussian',"OtsuThres"]
	for j in range(len(himg)):
		plt.subplot(3,3,j+1)
		plt.hist(himg[j].ravel(),256,[0,256])
		plt.title(htitles[j])
		plt.xticks([]),plt.yticks([])
def Ostu_thres_median(image_name,mavl,medpara):
	img = cv.imread(str(image_name),0)
	img_blur = cv.medianBlur(img,medpara)
	# find normalized_histogram, and its cumulative distribution function
	hist = cv.calcHist([img_blur],[0],None,[256],[0,256])
	hist_norm = hist.ravel()/hist.max()
	Q_hist = hist_norm.cumsum()
	bins = np.arange(256)
	fn_min = np.inf
	thresh = -1
	for i in range(1,256):
	    p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
	    q1,q2 = Q_hist[i],Q_hist[255]-Q_hist[i] # cum sum of classes
	    b1,b2 = np.hsplit(bins,[i]) # weights
	    # finding means and variances
	    m1,m2 = float(np.sum(p1*b1)/q1), float(np.sum(p2*b2)/q2)
	    v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2
	    # calculates the minimization function
	    fn = v1*q1 + v2*q2
	    if fn < fn_min:
	        fn_min = fn
	        thresh = i
	# find otsu's threshold value with OpenCV function
	ret, otsu = cv.threshold(img_blur,0,mavl,cv.THRESH_BINARY+cv.THRESH_OTSU)
	print( "thresh is {}".format(thresh) )

def compare_filter(image_name,vval,mavl,medpara,Gaussianparasize,Gaussianparacon,admean_block,Gaussian_block,c):
	img = cv.imread(str(image_name),0)
	img_blur_m = cv.medianBlur(img,medpara)
	img_blur_G = cv.GaussianBlur(img,(Gaussianparasize,Gaussianparasize),Gaussianparacon)
	titles = ['no filtering','Median filtering','Gaussian filtering']
	im = [img,img_blur_m,img_blur_G]
	for i in range(len(titles)):
		plt.figure()
		plt.hist(im[i].ravel(),256,[0,256])
		plt.suptitle(titles[i])
		plt.xticks([]),plt.yticks([])
def med_filtering_and_thresholding_stage2(image_name,vval,mavl,medpara,admean_block,Gaussian_block,c):
	img = cv.imread(str(image_name),0)
	img_blur = cv.medianBlur(img,medpara)
	titles = ['Original Image(blurred)','BINARY','TRUNC','TOZERO','AdaptMean', 'AdaptGaussian',"OtsuThres"]
	ret1,th1 = cv.threshold(img_blur,vval,mavl,cv.THRESH_BINARY)
	ret3,th2 = cv.threshold(img_blur,vval,mavl,cv.THRESH_TRUNC)
	ret4,th3 = cv.threshold(img_blur,vval,mavl,cv.THRESH_TOZERO)
	th4 = cv.adaptiveThreshold(img_blur,mavl,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,admean_block,c)
	th5 = cv.adaptiveThreshold(img_blur,mavl,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,Gaussian_block,c)
	ret6,th6 = cv.threshold(img_blur,0,mavl,cv.THRESH_BINARY+cv.THRESH_OTSU)
	plt.figure()
	plt.suptitle('Three typyes of Thresholding under Median filtering({})'.format(medpara))
	IMG = [img_blur,th1,th2,th3,th4,th5,th6]
	for i in range(len(IMG)):
		plt.subplot(3,3,i+1)
		plt.imshow(IMG[i],'gray')
		plt.title(titles[i])
		plt.xticks([]),plt.yticks([])
	plt.subplot(3,3,9)
	plt.title('Histogram')
	plt.hist(img_blur.ravel(),256,[0,256])
	plt.xticks([]),plt.yticks([])
def Gaussian_filtering_and_thresholding_stage2(image_name,vval,mavl,Gaussianparasize,Gaussianparacon,admean_block,Gaussian_block,c):
	img = cv.imread(str(image_name),0)
	img_blur = cv.GaussianBlur(img,(Gaussianparasize,Gaussianparasize),Gaussianparacon)
	titles = ['Original Image(blurred)','BINARY','TRUNC','TOZERO','AdaptMean', 'AdaptGaussian',"OtsuThres"]
	ret1,th1 = cv.threshold(img_blur,vval,mavl,cv.THRESH_BINARY)
	ret3,th2 = cv.threshold(img_blur,vval,mavl,cv.THRESH_TRUNC)
	ret4,th3 = cv.threshold(img_blur,vval,mavl,cv.THRESH_TOZERO)
	th4 = cv.adaptiveThreshold(img_blur,mavl,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,admean_block,c)
	th5 = cv.adaptiveThreshold(img_blur,mavl,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,Gaussian_block,c)
	ret6,th6 = cv.threshold(img_blur,0,mavl,cv.THRESH_BINARY+cv.THRESH_OTSU)
	plt.figure()
	plt.suptitle('Three typyes of Thresholding under Gaussian filtering({})'.format(Gaussianparasize))
	IMG = [img_blur,th1,th2,th3,th4,th5,th6]
	for i in range(len(IMG)):
		plt.subplot(3,3,i+1)
		plt.imshow(IMG[i],'gray')
		plt.title(titles[i])
		plt.xticks([]),plt.yticks([])
	plt.subplot(3,3,9)
	plt.title('Histogram')
	plt.hist(img_blur.ravel(),256,[0,256])
	plt.xticks([]),plt.yticks([])
def for_test(q1_1 = False,q1_2_median = False,q1_2_gaussian = False,q2_median = False,q2_gaussian = False):
	if q1_1:
		image_name_stage1= ['image1.png','image2.png']
		for i in image_name_stage1:
			sh = no_filtering_and_thresholding(i,127,255,25,35,2)
	if q1_2_median:
		image_name_stage1= ['image1.png','image2.png']
		size = [3,5,7,9,11]
		for i in image_name_stage1:
			for j in size:
				sh = med_filtering_and_thresholding(i,127,255,j,25,35,2)
	if q1_2_gaussian:
		image_name_stage1= ['image1.png','image2.png']
		size = [3,5,11,15,17]
		for i in image_name_stage1:
			for j in size:
				sh = Gaussian_filtering_and_thresholding(i,127,255,j,0,25,35,2) # standard deviation can be changed to test different outputs
	if q2_median:
		thre = [150,160,170]
		size = [3,5,7,9,11]
		for i in thre:
			for j in size:
				sh = med_filtering_and_thresholding_stage2('image3.jpg',i,255,j,25,35,2)
	if q2_gaussian:
		thre = [150,160,170]
		size = [3,5,11,15,17]
		for i in thre:
			for j in size:
				sh = Gaussian_filtering_and_thresholding_stage2('image3.jpg',i,255,j,0,25,35,2)


# we can select manipulation we want to test images
sh = for_test(q1_1 = False,q1_2_median = False,q1_2_gaussian = False,q2_median = False,q2_gaussian = False)

# return the threshold value calculated in Ostu thresholding algorithm with different kernel size of median filter which is the 3rd argument of this function
ss = Ostu_thres_median('image1.png',255,5)


# show the histograms of image1 and image2 under median filter and Gaussian filter
shown = compare_filter('image1.png',127,255,5,5,0,25,35,2)

plt.show()


















