


import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from datetime import datetime
from io import BytesIO

# =========================
# CONFIG
# =========================
# Use the light models (CHANGE 1)
yolo_model = YOLO("runs/detect/train3/weights/best.pt")
segmentation_model = YOLO("./segmentation_model/weights/best.pt")

# IP Webcam stream URL (prefer http over https)
# Example: "http://192.168.1.12:8080/video"
url = "http://10.107.191.165:8080//video"

# Frame skip settings (CHANGE 2)
SKIP_N = 5  # process 1 out of every 5 frames

# Other params
px_to_cm_ratio = 0.1
cap = cv2.VideoCapture(url)  # use phone stream instead of webcam 0

def load_models():
    """
    Load the small YOLO models (n) and MobileNetV2 classifier head.
    """
    models_status = {}
    try:
        # Force using small models for speed
        # yolo_model = YOLO("yolov8n.pt")
        models_status['yolo'] = "Using YOLOv8n (small) for detection"

        # segmentation_model = YOLO("yolov8n-seg.pt")
        models_status['segmentation'] = "Using YOLOv8n-seg (small) for segmentation"

        # Material Classification Model
        material_model = models.mobilenet_v2(weights='IMAGENET1K_V1')
        material_model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(material_model.last_channel, 8)  # 8 classes
        )
        material_model.eval()
        models_status['material'] = "MobileNetV2 with custom 8-class head loaded"

        return yolo_model, segmentation_model, material_model
    except Exception as e:
        print(f"❌ Model loading failed: {str(e)}")
        print("⚠ Some features may not work properly without models.")
        return None, None, None

# Load all models
yolo_model, segmentation_model, material_model = load_models()

def detect_biological_growth_advanced(image_np):
    try:
        growth_image = image_np.copy()
        hsv = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)

        lower_green1 = np.array([35, 40, 40])
        upper_green1 = np.array([85, 255, 255])
        lower_green2 = np.array([25, 30, 20])
        upper_green2 = np.array([95, 200, 150])

        mask_green1 = cv2.inRange(hsv, lower_green1, upper_green1)
        mask_green2 = cv2.inRange(hsv, lower_green2, upper_green2)

        combined_mask = cv2.bitwise_or(mask_green1, mask_green2)

        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        growth_detected = False
        total_growth_area = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:
                growth_detected = True
                cv2.drawContours(growth_image, [contour], -1, (0, 0, 255), 2)
                x, y, w, h = cv2.boundingRect(contour)
                cv2.putText(growth_image, f"{area:.0f}px",
                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                total_growth_area += area

        if not growth_detected:
            cv2.putText(growth_image,
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(growth_image, f"Total area: {total_growth_area:.0f} pixels",
                        (50, image_np.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return growth_image, growth_detected, total_growth_area
    except Exception as e:
        print(f"[ERROR] Biological growth detection failed: {e}")
        return image_np, False, 0

def classify_material(image_np):
    try:
        if material_model is None:
            return classify_material_fallback(image_np)

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image_tensor = transform(image_rgb).unsqueeze(0)

        with torch.no_grad():
            output = material_model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)

        material_classes = ['Stone', 'Brick', 'Plaster', 'Concrete', 'Wood', 'Metal', 'Marble', 'Sandstone']
        predicted_material = material_classes[predicted.item()]
        probs = probabilities[0].cpu().numpy()

        return predicted_material, probs
    except Exception as e:
        print(f"[ERROR] Material classification failed: {e}")
        return classify_material_fallback(image_np)

def classify_material_fallback(image_np):
    try:
        hsv = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
        mean_hue = np.mean(hsv[:, :, 0])
        mean_saturation = np.mean(hsv[:, :, 1])
        mean_value = np.mean(hsv[:, :, 2])
        std_value = np.std(hsv[:, :, 2])
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        texture_measure = np.std(gray)
        mean_b, mean_g, mean_r = np.mean(image_np, axis=(0, 1))

        if mean_r > mean_g > mean_b and mean_saturation > 80:
            material = 'Brick'; probs = np.array([0.1, 0.7, 0.05, 0.05, 0.05, 0.02, 0.02, 0.01])
        elif texture_measure > 60 and mean_value < 120:
            if mean_value < 80:
                material = 'Stone'; probs = np.array([0.8, 0.05, 0.05, 0.05, 0.02, 0.01, 0.01, 0.01])
            else:
                material = 'Sandstone'; probs = np.array([0.2, 0.05, 0.05, 0.05, 0.02, 0.01, 0.1, 0.6])
        elif mean_value > 180 and std_value < 30:
            if mean_saturation < 20:
                if texture_measure < 20:
                    material = 'Marble'; probs = np.array([0.05, 0.05, 0.1, 0.05, 0.02, 0.01, 0.7, 0.02])
                else:
                    material = 'Plaster'; probs = np.array([0.05, 0.1, 0.7, 0.05, 0.05, 0.02, 0.02, 0.01])
            else:
                material = 'Concrete'; probs = np.array([0.1, 0.05, 0.1, 0.6, 0.05, 0.05, 0.03, 0.02])
        elif 10 < mean_hue < 30 and mean_saturation > 50:
            material = 'Wood'; probs = np.array([0.05, 0.1, 0.05, 0.05, 0.7, 0.02, 0.02, 0.01])
        elif mean_value > 150 and texture_measure > 40:
            if mean_saturation < 30:
                material = 'Metal'; probs = np.array([0.02, 0.05, 0.05, 0.1, 0.05, 0.7, 0.02, 0.01])
            else:
                material = 'Concrete'; probs = np.array([0.1, 0.05, 0.1, 0.6, 0.05, 0.05, 0.03, 0.02])
        else:
            material = 'Stone'; probs = np.array([0.5, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05])

        return material, probs
    except Exception as e:
        print(f"Fallback material classification failed: {str(e)}")
        return 'Unknown', np.array([0.125] * 8)

def visualize_material_classification(image_np, material, probabilities):
    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        materials = ['Stone', 'Brick', 'Plaster', 'Concrete', 'Wood', 'Metal', 'Marble', 'Sandstone']
        colors = ['#8B4513', '#FF4500', '#FFD700', '#808080', '#DEB887', '#C0C0C0', '#F5F5DC', '#F4A460']
        bars = ax.bar(materials, probabilities, color=colors)
        ax.set_title(f'Material Classification: {material}', fontsize=16, fontweight='bold')
        ax.set_ylabel('Confidence Score', fontsize=12)
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=45)
        for bar, prob in zip(bars, probabilities):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.01, f'{prob:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        image = np.array(Image.open(buf))
        plt.close(fig)
        return image
    except Exception as e:
        print(f"Material visualization failed: {str(e)}")
        return image_np

def detect_biological_growth(image_np, crack_details):
    try:
        growth_image, growth_detected_advanced, growth_area_px = detect_biological_growth_advanced(image_np)
        yolo_growth_detected = False
        for crack in crack_details:
            if any(k in crack['label'].lower() for k in ['moss', 'growth', 'algae', 'lichen', 'vegetation']):
                x1, y1, x2, y2 = crack['bbox']
                width_cm = crack['width_cm']; length_cm = crack['length_cm']; confidence = crack['confidence']
                cv2.rectangle(growth_image, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.putText(growth_image, f"YOLO Growth: {width_cm:.2f}cm x {length_cm:.2f}cm ({confidence:.2f})",
                            (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                yolo_growth_detected = True

        if growth_detected_advanced or yolo_growth_detected:
            cv2.putText(growth_image, " BIOLOGICAL GROWTH DETECTED ",
                        (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        return growth_image
    except Exception as e:
        print(f"[ERROR] Biological growth detection failed: {e}")
        return image_np

def calculate_biological_growth_area(crack_details, seg_results, image_np, px_to_cm_ratio):
    try:
        total_area_cm2 = 0
        for crack in crack_details:
            if any(k in crack['label'].lower() for k in ['moss', 'growth', 'algae', 'lichen', 'vegetation']):
                total_area_cm2 += crack['width_cm'] * crack['length_cm']

        _, growth_detected, growth_area_px = detect_biological_growth_advanced(image_np)
        if growth_detected and growth_area_px > 0:
            total_area_cm2 += growth_area_px * (px_to_cm_ratio ** 2)

        if seg_results and hasattr(seg_results[0], 'masks') and seg_results[0].masks is not None:
            masks = seg_results[0].masks.data.cpu().numpy()
            h, w = image_np.shape[:2]
            for mask in masks:
                resized_mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                total_area_cm2 += np.sum(resized_mask) * (px_to_cm_ratio ** 2)
        return total_area_cm2
    except Exception as e:
        print(f"[ERROR] Biological growth area calc failed: {e}")
        return 0.0

def preprocess_image_for_depth_estimation(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    equalized = cv2.equalizeHist(blurred)
    return equalized

def create_depth_estimation_heatmap(equalized):
    _, mask = cv2.threshold(equalized, 60, 255, cv2.THRESH_BINARY_INV)
    shadow = cv2.bitwise_and(equalized, equalized, mask=mask)
    depth = 255 - shadow
    norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)

def apply_canny(image_np):
    return cv2.Canny(image_np, 100, 200)

def detect_with_yolo(image_np):
    results = yolo_model(image_np)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            w, h = (x2 - x1), (y2 - y1)
            label = yolo_model.names[int(box.cls[0])]
            conf = float(box.conf[0].cpu().numpy())
            txt = f"{label} {w*px_to_cm_ratio:.1f}cm x {h*px_to_cm_ratio:.1f}cm ({conf:.2f})"
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(image_np, txt, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
    return results[0].plot()

def segment_image(image_np):
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    results = segmentation_model.predict(source=image_rgb, conf=0.3, save=False)
    return results[0].plot()

print("Press 'q' to quit.")

frame_count = 0
last_dashboard = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame from stream. Check the URL (use http).")
        break

    frame = cv2.resize(frame, (640, 480))
    frame_count += 1

    # =========================
    # FRAME SKIP (CHANGE 2)
    # =========================
    if frame_count % SKIP_N != 0 and last_dashboard is not None:
        cv2.imshow("Heritage Health Monitoring Dashboard", last_dashboard)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Build crack details from detection pass
    results = yolo_model(frame.copy())
    crack_details = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            w, h = (x2 - x1), (y2 - y1)
            label = yolo_model.names[int(box.cls[0])]
            conf = float(box.conf[0].cpu().numpy())
            crack_details.append({
                'label': label,
                'bbox': (x1, y1, x2, y2),
                'width_cm': w * px_to_cm_ratio,
                'length_cm': h * px_to_cm_ratio,
                'confidence': conf
            })

    detection = detect_with_yolo(frame.copy())
    segmentation = segment_image(frame.copy())
    depth = create_depth_estimation_heatmap(preprocess_image_for_depth_estimation(frame.copy()))
    edges = cv2.cvtColor(apply_canny(frame.copy()), cv2.COLOR_GRAY2BGR)
    growth_image = detect_biological_growth(frame.copy(), crack_details)
    material, probabilities = classify_material(frame.copy())
    material_viz = visualize_material_classification(frame.copy(), material, probabilities)

    standard_size = (640, 480)
    detection = cv2.resize(detection, standard_size)
    segmentation = cv2.resize(segmentation, standard_size)
    depth = cv2.resize(depth, standard_size)
    edges = cv2.resize(edges, standard_size)
    growth_image = cv2.resize(growth_image, standard_size)
    material_viz = cv2.resize(material_viz, standard_size)

    if material_viz.shape[2] == 4:
        material_viz = cv2.cvtColor(material_viz, cv2.COLOR_RGBA2BGR)

    top_row = np.hstack((detection, segmentation, growth_image))
    bottom_row = np.hstack((depth, edges, material_viz))
    dashboard = np.vstack((top_row, bottom_row))

    last_dashboard = dashboard  # cache for skipped frames
    cv2.imshow("Heritage Health Monitoring Dashboard", dashboard)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
