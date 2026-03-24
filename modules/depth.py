import cv2
import numpy as np

def compute_disparity_map(img_left, img_right):
    if len(img_left.shape) == 3:
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    else:
        gray_left = img_left
        
    if len(img_right.shape) == 3:
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    else:
        gray_right = img_right
        
    # Resize right image to match left if needed (stereo pair should be same size)
    if gray_left.shape != gray_right.shape:
        gray_right = cv2.resize(gray_right, (gray_left.shape[1], gray_left.shape[0]))
        
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(gray_left, gray_right)
    
    # Normalize disparity for visualization
    disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    return disparity_normalized
