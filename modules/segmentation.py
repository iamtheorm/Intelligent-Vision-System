import cv2
import numpy as np

def apply_kmeans_segmentation(image, k=3):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Reshape the image to a 2D array of pixels
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    
    # Define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert back to 8-bit values
    centers = np.uint8(centers)
    
    # Map labels to center colors
    segmented_image = centers[labels.flatten()]
    
    # Reshape back to the original image dimensions
    segmented_image = segmented_image.reshape(image.shape)
    
    return segmented_image
