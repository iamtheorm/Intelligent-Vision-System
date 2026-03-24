import cv2

def process_motion_frame(frame, backSub):
    """
    Applies background subtraction to a single frame and draws bounding boxes
    around detected motion contours.
    Returns the processed frame, fgMask, and a count of moving objects.
    """
    fgMask = backSub.apply(frame)
    
    # Optional: Apply some morphology to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    motion_count = 0
    output_frame = frame.copy()
    for contour in contours:
        if cv2.contourArea(contour) > 500: # Threshold for minimum area
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            motion_count += 1
            
    return output_frame, fgMask, motion_count

def get_background_subtractor():
    return cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
