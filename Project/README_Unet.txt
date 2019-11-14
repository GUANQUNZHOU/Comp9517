##############################
####### Deep Learning ########
##############################

Code filename: "DL_Unet_ResUnet.ipynb"
Code Description: Code used for Implementing Unet and ResUnet

code for deep learning models were developed and tested on Kaggle Kernels using Keras.
Libariy Requirment:
1. tensorflow
2. keras
3. tifffile
4. skimage
5. Scikit-learn
6. numpy

Input Data set Requirments:
1. As we are developing our code using Kaggle Kernel, we have uploaded give image and label files onto the Kernel Workspace.
2. If you want to run the code, ensure to change the "imagePath" and "labelPath" variables to corresponding path.

Model selection:
Two methods have successfully been implemented , namely U-net and U-net++
To select a specific model to run, simply uncomment the correspond code blocks and comment out the other one

Output:
After successfully running the code, 3 files are generated:
	- one .zip file which contains all 30 output images
	- one .tif file which stacks all 30 .tif images into one large .tif file 
	- one .hdf5 file which stores model weights