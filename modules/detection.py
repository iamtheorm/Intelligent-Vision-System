import cv2

def detect_faces(image):
    # Load the pre-trained Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Ensure it's in grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw rectangles dynamically
    output_image = image.copy()
    
    # If original image was grayscale, convert output to BGR so we can draw colored boxes
    if len(output_image.shape) == 2:
        output_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2BGR)
        
    for (x, y, w, h) in faces:
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    return output_image, faces
