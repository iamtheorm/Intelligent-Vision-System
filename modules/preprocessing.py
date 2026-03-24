import cv2
import numpy as np

def apply_gaussian_blur(image, kernel_size=(5, 5)):
    return cv2.GaussianBlur(image, kernel_size, 0)

def apply_histogram_equalization(image):
    if len(image.shape) == 3:
        # Convert to YUV, equalize Y channel, convert back to BGR
        yuv_img = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        yuv_img[:, :, 0] = cv2.equalizeHist(yuv_img[:, :, 0])
        return cv2.cvtColor(yuv_img, cv2.COLOR_YUV2BGR)
    else:
        return cv2.equalizeHist(image)

def apply_image_sharpening(image):
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)
