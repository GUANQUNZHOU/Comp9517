# Template for computing SIFT features

import cv2
import numpy as np
from matplotlib import pyplot as plt


class SiftDetector():
    def __init__(self, norm="L2", params=None):
        self.detector=self.get_detector(params)
        self.norm=norm

    def get_detector(self, params):
        if params is None:
            params={}
            params["n_features"]=0
            params["n_octave_layers"]=3
            params["contrast_threshold"]=0.04
            params["edge_threshold"]=10
            params["sigma"]=1.6
        detector = cv2.xfeatures2d.SIFT_create(
                nfeatures=params["n_features"],
                nOctaveLayers=params["n_octave_layers"],
                contrastThreshold=params["contrast_threshold"],
                edgeThreshold=params["edge_threshold"],
                sigma=params["sigma"])

        return detector

def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

if __name__ == '__main__':
    # 1. Read the colour image
    img = cv2.imread('NotreDame.jpg')
    # For task 2 only, rotate the image by 45 degrees
    img_2 = rotate(img, -45)
    # 2. Convert image to greyscale	
    gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_2= cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
    # 3. Initialise SIFT detector (with varying parameters)
    params={}
    params["n_features"]=0
    params["n_octave_layers"]=3
    params["contrast_threshold"]=0.1
    params["edge_threshold"]=10
    params["sigma"]=1.6
    b = SiftDetector(params = params)
    
    # 4. Detect and compute SIFT features
    kp = b.detector.detect(gray,None)
    kps, features = b.detector.detectAndCompute(gray, None)
    print(features)
    print(len(features))
    kp_2 = b.detector.detect(gray_2,None)
    cv2.drawKeypoints(img,kp, img)#flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.drawKeypoints(img_2,kp_2, img_2)#flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # 5. Visualise detected keypoints on the colour image 
    cv2.imwrite('Task1.jpg',img)
    cv2.imwrite('Task2.jpg',img_2)
