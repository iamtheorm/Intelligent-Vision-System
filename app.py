import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

from modules import preprocessing, feature_extraction, segmentation, detection, classification, motion, depth, shape

st.set_page_config(page_title="Intelligent Vision System", layout="wide")
st.title("Intelligent Vision System 👁️")
st.sidebar.title("Navigation")
st.sidebar.markdown("Choose a computer vision module:")

app_mode = st.sidebar.selectbox("Select Module", [
    "Introduction",
    "Image Processing",
    "Feature Detection",
    "Segmentation",
    "Object Detection",
    "Classification",
    "Motion Tracking",
    "Depth Estimation",
    "Shape from Shading"
])

def load_image():
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_cv = np.array(image.convert('RGB'))
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
        return image, image_cv
    return None, None

def load_two_images():
    col1, col2 = st.columns(2)
    with col1:
        file1 = st.file_uploader("Upload Left Image", type=["jpg", "png"], key="left")
    with col2:
        file2 = st.file_uploader("Upload Right Image", type=["jpg", "png"], key="right")
        
    img1_cv, img2_cv = None, None
    if file1 and file2:
        img1 = Image.open(file1)
        img1_cv = cv2.cvtColor(np.array(img1.convert('RGB')), cv2.COLOR_RGB2BGR)
        img2 = Image.open(file2)
        img2_cv = cv2.cvtColor(np.array(img2.convert('RGB')), cv2.COLOR_RGB2BGR)
    return img1_cv, img2_cv

if app_mode == "Introduction":
    st.markdown("""
    Welcome to the **Intelligent Vision System**!
    This application demonstrates a complete computer vision pipeline using Streamlit and OpenCV.
    Please use the sidebar to navigate through different modules.
    """)

elif app_mode == "Image Processing":
    st.header("Image Processing Module")
    img_pil, img_cv = load_image()
    if img_cv is not None:
        st.image(img_pil, caption="Original Image", use_container_width=True)
        process_type = st.radio("Select Processing Type", ["Gaussian Blur", "Histogram Equalization", "Image Sharpening"])
        
        if st.button("Apply"):
            if process_type == "Gaussian Blur":
                res = preprocessing.apply_gaussian_blur(img_cv, (5, 5))
            elif process_type == "Histogram Equalization":
                res = preprocessing.apply_histogram_equalization(img_cv)
            else:
                res = preprocessing.apply_image_sharpening(img_cv)
                
            res_rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
            st.image(res_rgb, caption=f"Result: {process_type}", use_container_width=True)

elif app_mode == "Feature Detection":
    st.header("Feature Extraction Module")
    img_pil, img_cv = load_image()
    if img_cv is not None:
        st.image(img_pil, caption="Original Image", use_container_width=True)
        feature_type = st.radio("Select Feature Type", ["Canny Edge Detection", "HOG Feature Map"])
        
        if st.button("Extract"):
            if feature_type == "Canny Edge Detection":
                res = feature_extraction.apply_canny_edge(img_cv)
                st.image(res, caption="Canny Edges", use_container_width=True)
            else:
                _, res = feature_extraction.apply_hog(img_cv)
                st.image(res, caption="HOG Features", use_container_width=True)

elif app_mode == "Segmentation":
    st.header("Image Segmentation Module")
    img_pil, img_cv = load_image()
    if img_cv is not None:
        st.image(img_pil, caption="Original Image", use_container_width=True)
        k = st.slider("Select K for K-Means", 2, 10, 3)
        if st.button("Segment"):
            res = segmentation.apply_kmeans_segmentation(img_cv, k)
            res_rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
            st.image(res_rgb, caption=f"K-Means Segmented (K={k})", use_container_width=True)

elif app_mode == "Object Detection":
    st.header("Object Detection Module (Faces)")
    img_pil, img_cv = load_image()
    if img_cv is not None:
        st.image(img_pil, caption="Original Image", use_container_width=True)
        if st.button("Detect Faces"):
            res, faces = detection.detect_faces(img_cv)
            res_rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
            st.image(res_rgb, caption=f"Detected {len(faces)} faces", use_container_width=True)

elif app_mode == "Classification":
    st.header("Classification Module (KNN)")
    st.write("Train a simple dummy K-Nearest Neighbors classifier and test it.")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Train KNN Dummy Model"):
            msg = classification.train_knn_dummy()
            st.success(msg)
            
    with col2:
        st.write("Test the model with a 2D feature vector:")
        val1 = st.number_input("Feature 1", value=0.0)
        val2 = st.number_input("Feature 2", value=0.0)
        if st.button("Classify"):
            res = classification.classify_dummy_feature([val1, val2])
            st.info(res)

elif app_mode == "Motion Tracking":
    st.header("Motion Tracking Module")
    st.write("Upload a video to track motion via Background Subtraction.")
    video_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    
    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        backSub = motion.get_background_subtractor()
        
        if st.button("Start Tracking"):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                out_frame, _, _ = motion.process_motion_frame(frame, backSub)
                out_frame_rgb = cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB)
                stframe.image(out_frame_rgb, use_container_width=True)
            cap.release()
            os.unlink(tfile.name)

elif app_mode == "Depth Estimation":
    st.header("Depth Estimation Module (Stereo Vision)")
    st.write("Upload left and right stereo images to compute the disparity map.")
    img_left, img_right = load_two_images()
    
    if img_left is not None and img_right is not None:
        if st.button("Compute Disparity"):
            disp = depth.compute_disparity_map(img_left, img_right)
            st.image(disp, caption="Disparity Map", use_container_width=True)

elif app_mode == "Shape from Shading":
    st.header("Shape from Shading Module")
    img_pil, img_cv = load_image()
    if img_cv is not None:
        st.image(img_pil, caption="Original Image", use_container_width=True)
        if st.button("Estimate Shape Depth"):
            res = shape.shape_from_shading(img_cv)
            st.image(res, caption="Shape/Depth Approximation", use_container_width=True)
