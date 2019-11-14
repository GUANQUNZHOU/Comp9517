##############################
####### Machine Learning ########
##############################

Code filename: "Feature_extraction_and_SVM.ipynb"
Code Description: Code used for Implementing Feature extraction+SVM

Our code for machine learning models were developed and tested on Jupiter Notebooks.
Libariy Requirment:
1. Pandas
2. Numpy
3. tifffile
4. skimage
5. Scikit-learn
6. scipy
7.itertools
8.matplotlib
Input Data set Requirments:
1. As the images and labels provided are .jgp files, the code is based on .jpg files
2. If you want to run the code, ensure to change the "imagePath" and "labelPath" variables to corresponding path.

Model selection:
This code file is specifically for feature_extraction+SVM model.

Code implementation:
In this code, there are three main stages being implemented:
	(a) preprocessing: code achieves threshold and blob detection.
	(b) Feature extraction: code achieves to extract features for each image and  produce a vector based on each set of features.
	(c) classification: code implements SVC model to complete training and prediction.

This code also improves performance by using image patching.

Output:
After successfully running the code, one .tif file which stacks all 30 .tif images into one large .tif file have been generated.

Note:
Code needs time to run and output(about 3-4 hours).