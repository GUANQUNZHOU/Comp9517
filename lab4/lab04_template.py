"""
COMP9517 Lab 04, Week 6
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from sklearn.cluster import MeanShift

from PIL import Image

size = 100, 100

img_dir = "data"
ext_dir = "ext_data"

img_names = ["orange_half.png", "two_halves_binary.png"]
ext_names = ["coins.png", "two_halves.png"]

images = [img_dir + "/" + i for i in img_names]
ext_images = [ext_dir + "/" + i for i in ext_names]


def plot_three_images(figure_title, image1, label1,
                      image2, label2, image3, label3):
    fig = plt.figure()
    fig.suptitle(figure_title)

    # Display the first image
    fig.add_subplot(1, 3, 1)
    plt.imshow(image1)
    plt.axis('off')
    plt.title(label1)

    # Display the second image
    fig.add_subplot(1, 3, 2)
    plt.imshow(image2)
    plt.axis('off')
    plt.title(label2)

    # Display the third image
    fig.add_subplot(1, 3, 3)
    plt.imshow(image3)
    plt.axis('off')
    plt.title(label3)

    plt.show()

for img_path in images:
    img = Image.open(img_path)
    img.thumbnail(size)  # Convert the image to 100 x 100
    # Convert the image to a numpy matrix
    img_mat = np.array(img)[:, :, :3]
    
    # +--------------------+
    # |     Question 1     |
    # +--------------------+
    #
    # TODO: perform MeanShift on image
    # Follow the hints in the lab spec.

    # Step 1 - Extract the three RGB colour channels
    # Hint: It will be useful to store the shape of one of the colour
    # channels so we can reshape the flattened matrix back to this shape.
    R = img_mat[:,:,0]
    G = img_mat[:,:,1]
    B = img_mat[:,:,2]
    form = R.shape
    # Step 2 - Combine the three colour channels by flatten each channel 
	# then stacking the flattened channels together.
    # This gives the "colour_samples"
    R_f = R.flatten()
    G_f = G.flatten()
    B_f = B.flatten()
    colour_samples = np.array([R_f,G_f,B_f]).T
    # Step 3 - Perform Meanshift  clustering
    #
    ms_clf = MeanShift(bin_seeding=True)
    ms_labels = ms_clf.fit_predict(colour_samples)
    ms_labels = np.reshape(ms_labels,form)
    # print(ms_labels)
    # Step 4 - reshape ms_labels back to the original image shape 
	# for displaying the segmentation output
    #%%
    #
    # +--------------------+
    # |     Question 2     |
    # +--------------------+
    #

    # TODO: perform Watershed on image
    # Follow the hints in the lab spec.

    # Step 1 - Convert the image to gray scale
    # and convert the image to a numpy matrix
    gray = img.convert('L')
    # gray = cv2.imread(img_path,0)
    img_array = np.array(gray)
    # Step 2 - Calculate the distance transform
    # Hint: use     ndi.distance_transform_edt(img_array)
    distance = ndi.distance_transform_edt(img_array)
    # Step 3 - Generate the watershed markers
    # Hint: use the peak_local_max() function from the skimage.feature library
    # to get the local maximum values and then convert them to markers
    # using ndi.label() -- note the markers are the 0th output to this function
    local_max = peak_local_max(distance,indices=False, footprint=np.ones((3, 3)),labels=img_array)
    markers = ndi.label(local_max)[0]
    # Step 4 - Perform watershed and store the labels
    # Hint: use the watershed() function from the skimage.morphology library
    # with three inputs: -distance, markers and your image array as a mask
    ws_labels = watershed(-distance, markers,mask=img_array)
    # Display the results 
    plot_three_images(img_path, img, "Original Image", ms_labels, "MeanShift Labels",
                      ws_labels, "Watershed Labels")

    # If you want to visualise the watershed distance markers then try
    # plotting the code below.
    # plot_three_images(img_path, img, "Original Image", -distance, "Watershed Distance",
    #                   ws_labels, "Watershed Labels")

#%%
#
# +-------------------+
# |     Extension     |
# +-------------------+
#
# Loop for the extension component
for img_path in ext_images:
    img = Image.open(img_path)
    img.thumbnail(size)
    img_ex = np.array(img)[:, :, :3]

    # TODO: perform meanshift on image
    R_ex = img_ex[:,:,0]
    G_ex = img_ex[:,:,1]
    B_ex = img_ex[:,:,2]
    form_ex = R_ex.shape
    R_f_ex = R_ex.flatten()
    G_f_ex = G_ex.flatten()
    B_f_ex = B_ex.flatten()
    colour_samples_ex = np.array([R_f_ex,G_f_ex,B_f_ex]).T
    ms_clf = MeanShift(bin_seeding=True)
    ms_labels = ms_clf.fit_predict(colour_samples_ex)
    ms_labels = np.reshape(ms_labels,form_ex)

    # TODO: perform an optimisation and then watershed on image

    img_e = cv2.imread(img_path)
    gray_ex = cv2.cvtColor(img_e,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_ex,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2) 

    sure_bg = cv2.dilate(opening,kernel,iterations=3)

    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)
     
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)

    markers = markers+1

    markers[unknown==255] = 0
    ws_labels = img_e
    markers = cv2.watershed(ws_labels,markers)
    ws_labels[markers == -1] = [255,0,0]
    plot_three_images(img_path, img, "Original Image", ms_labels, "MeanShift Labels",
                      ws_labels, "Watershed Labels")
