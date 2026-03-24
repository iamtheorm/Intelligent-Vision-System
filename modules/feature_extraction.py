import cv2
import numpy as np

def apply_canny_edge(image, threshold1=100, threshold2=200):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return cv2.Canny(gray, threshold1, threshold2)

def apply_hog(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Initialize HOG descriptor
    hog = cv2.HOGDescriptor()
    
    # Resize to standard HOG window size to extract features properly
    img_resized = cv2.resize(gray, (64, 128))
    features = hog.compute(img_resized)
    
    # For visualization, show the gradient magnitude which is the basis of HOG
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    hog_image_rescaled = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    return features, hog_image_rescaled
