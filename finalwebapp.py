# THE ORIGINAL ONE(only one image)

# import streamlit as st
# import torch
# import cv2
# import numpy as np
# from PIL import Image
# import time
# from ultralytics import YOLO

# st.set_page_config(page_title="Heritage Sites Health Monitoring", layout="wide")


# yolo_model_path = "runs/detect/train3/weights/best.pt"
# yolo_model = YOLO(yolo_model_path)


# segmentation_model_path = "./segmentation_model/weights/best.pt"
# segmentation_model = YOLO(segmentation_model_path)


# st.sidebar.header("Options")


# model_choice = st.sidebar.selectbox("Select a model for object detection:", ("YOLO",))


# uploaded_file = st.sidebar.file_uploader(
#     "Choose an image...", type=["jpg", "png", "jpeg"]
# )


# px_to_cm_ratio = 0.1


# def preprocess_image_for_depth_estimation(image_np):
#     gray_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
#     blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
#     equalized_image = cv2.equalizeHist(blurred_image)
#     return equalized_image


# def create_depth_estimation_heatmap(equalized_image):
#     _, shadow_mask = cv2.threshold(equalized_image, 60, 255, cv2.THRESH_BINARY_INV)
#     shadow_region = cv2.bitwise_and(equalized_image, equalized_image, mask=shadow_mask)
#     depth_estimation = 255 - shadow_region
#     depth_estimation_normalized = cv2.normalize(
#         depth_estimation, None, 0, 255, cv2.NORM_MINMAX
#     )
#     depth_heatmap_colored = cv2.applyColorMap(
#         depth_estimation_normalized.astype(np.uint8), cv2.COLORMAP_JET
#     )
#     return depth_heatmap_colored


# def apply_canny_edge_detection(image_np):
#     edges = cv2.Canny(image_np, 100, 200)
#     return edges


# def detect_with_yolo(image_np):
#     results = yolo_model(image_np)
#     for result in results:
#         boxes = result.boxes
#         for box in boxes:
#             x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
#             width_px = x2 - x1
#             height_px = y2 - y1
#             width_cm = width_px * px_to_cm_ratio
#             height_cm = height_px * px_to_cm_ratio

#             class_id = int(box.cls[0].cpu().numpy())
#             confidence = box.conf[0].cpu().numpy()
#             label = yolo_model.names[class_id]

#             # Draw the bounding box and label on the image
#             cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             dimension_text = (
#                 f"         {width_cm:.2f}cm x {height_cm:.2f}cm ({confidence:.2f})"
#             )
#             cv2.putText(
#                 image_np,
#                 dimension_text,
#                 (x1, y1 - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.5,
#                 (255, 0, 0),
#                 2,
#             )

#     annotated_image = results[0].plot()
#     return annotated_image


# def segment_image(image_np):

#     image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

#     results = segmentation_model.predict(source=image_rgb, conf=0.3, save=False)

#     segmented_image = results[0].plot()

#     return segmented_image


# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     image_np = np.array(image)
#     image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

#     st.subheader("Uploaded Image")
#     st.image(image, caption="Uploaded Image", use_container_width=True)

#     with st.spinner("Analyzing the image..."):
#         time.sleep(2)

#         processed_image = detect_with_yolo(image_np)
#         image = Image.open(uploaded_file)
#         image_np = np.array(image)
#         image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

#         segmented_image = segment_image(image_np)

#         image = Image.open(uploaded_file)
#         image_np = np.array(image)
#         image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

#         equalized_image = preprocess_image_for_depth_estimation(image_np)
#         depth_heatmap = create_depth_estimation_heatmap(equalized_image)
#         edges = apply_canny_edge_detection(image_np)

#     col1, col2 = st.columns(2)
#     with col1:
#         st.image(
#             processed_image,
#             caption=f"Detection Results using {model_choice}",
#             use_container_width=True,
#         )
#         st.image(
#             depth_heatmap, caption="Depth Estimation Heatmap", use_container_width=True
#         )
#     with col2:
#         st.image(segmented_image, caption="Segmentation Result", use_container_width=True)
#         st.image(edges, caption="Canny Edge Detection", use_container_width=True)


# st.markdown(
#     '<div class="footer">© 2024 Heritage Health Monitoring <i class="fas fa-globe"></i></div>',
#     unsafe_allow_html=True,
# )









import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import time
import tempfile
from ultralytics import YOLO

st.set_page_config(page_title="Heritage Sites Health Monitoring", layout="wide")

yolo_model_path = "runs/detect/train3/weights/best.pt"
yolo_model = YOLO(yolo_model_path)

segmentation_model_path = "./segmentation_model/weights/best.pt"
segmentation_model = YOLO(segmentation_model_path)

st.sidebar.header("Options")

model_choice = st.sidebar.selectbox("Select a model for object detection:", ("YOLO",))

uploaded_files = st.sidebar.file_uploader(
    "Choose images...", type=["jpg", "png", "jpeg"], accept_multiple_files=True
)

uploaded_video = st.sidebar.file_uploader("Upload a video...",type=["mp4","avi","mov"])

px_to_cm_ratio = 0.1

def preprocess_image_for_depth_estimation(image_np):
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    equalized_image = cv2.equalizeHist(blurred_image)
    return equalized_image

def create_depth_estimation_heatmap(equalized_image):
    _, shadow_mask = cv2.threshold(equalized_image, 60, 255, cv2.THRESH_BINARY_INV)
    shadow_region = cv2.bitwise_and(equalized_image, equalized_image, mask=shadow_mask)
    depth_estimation = 255 - shadow_region
    depth_estimation_normalized = cv2.normalize(
        depth_estimation, None, 0, 255, cv2.NORM_MINMAX
    )
    depth_heatmap_colored = cv2.applyColorMap(
        depth_estimation_normalized.astype(np.uint8), cv2.COLORMAP_JET
    )
    return depth_heatmap_colored

def apply_canny_edge_detection(image_np):
    edges = cv2.Canny(image_np, 100, 200)
    return edges

def detect_with_yolo(image_np):
    results = yolo_model(image_np)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            width_px = x2 - x1
            height_px = y2 - y1
            width_cm = width_px * px_to_cm_ratio
            height_cm = height_px * px_to_cm_ratio

            class_id = int(box.cls[0].cpu().numpy())
            confidence = box.conf[0].cpu().numpy()
            label = yolo_model.names[class_id]

            # Draw the bounding box and label on the image
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            dimension_text = (
                f"         {width_cm:.2f}cm x {height_cm:.2f}cm ({confidence:.2f})"
            )
            cv2.putText(
                image_np,
                dimension_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )

    annotated_image = results[0].plot()
    return annotated_image

def segment_image(image_np):
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    results = segmentation_model.predict(source=image_rgb, conf=0.3, save=False)
    segmented_image = results[0].plot()
    return segmented_image

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        st.subheader(f"Uploaded Image: {uploaded_file.name}")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Analyzing the image..."):
            time.sleep(2)

            processed_image = detect_with_yolo(image_np)
            segmented_image = segment_image(image_np)
            equalized_image = preprocess_image_for_depth_estimation(image_np)
            depth_heatmap = create_depth_estimation_heatmap(equalized_image)
            edges = apply_canny_edge_detection(image_np)

        col1, col2 = st.columns(2)
        with col1:
            st.image(processed_image,caption=f"Detection Results using {model_choice}",use_container_width=True,)
            st.image(depth_heatmap, caption="Depth Estimation Heatmap", use_container_width=True)
        with col2:
            st.image(segmented_image, caption="Segmentation Result", use_container_width=True)
            st.image(edges, caption="Canny Edge Detection", use_container_width=True)
        
        st.markdown("---")

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete = False)
    tfile.write(uploaded_video.read())
    cap = cv2.VideoCapture(tfile.name)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps)
    frame_idx = 0
    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read() 
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            frame_number += 1
            st.subhead(f"Frame {frame_number}")

            image_np = frame
            processed_image = detect_with_yolo(image_np)
            segmented_image = segment_image(image_np)
            equalized_image = preprocess_image_for_depth_estimation(image_np)
            depth_heatmap = create_depth_estimation_heatmap(equalized_image)
            edges = apply_canny_edge_detection(image_np)

            col1,col2 = st.columns(2)
            with col1:
                st.image(processed_image, caption="Detection",use_container_width = True)
                st.image(depth_heatmap,caption="Depth Estimation",use_container_width=True)
            with col2:
                st.image(segment_image,caption="Segmentaion",use_container_width=True)
                st.image(edges,caption="Edges",use_container_width=True)
            st.markdown("---")
        frame_idx += 1
    cap.release()

st.markdown(
    '<div class="footer">© 2024 Heritage Health Monitoring <i class="fas fa-globe"></i></div>',
    unsafe_allow_html=True,
)