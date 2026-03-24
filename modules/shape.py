import cv2
import numpy as np

def shape_from_shading(image):
    """
    A very basic approximation to show depth/surface variation using grayscale gradients.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    # Calculate gradients
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Combine gradients to get a pseudo-depth map (magnitude)
    magnitude = cv2.magnitude(grad_x, grad_y)
    
    # Normalize
    depth_map = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    return depth_map
