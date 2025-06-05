import cv2
import numpy as np
from ultralytics import YOLO

# Load models
yolo_model = YOLO("runs/detect/train3/weights/best.pt")
segmentation_model = YOLO("segmentation_model/weights/best.pt")

# Parameters
px_to_cm_ratio = 0.1
cap = cv2.VideoCapture(0)  # Use webcam

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
            conf = box.conf[0].cpu().numpy()
            txt = f"{label} {w*px_to_cm_ratio:.1f}cm x {h*px_to_cm_ratio:.1f}cm ({conf:.2f})"
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(image_np, txt, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
    return results[0].plot()

def segment_image(image_np):
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    results = segmentation_model.predict(source=image_rgb, conf=0.3, save=False)
    return results[0].plot()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for consistency (optional)
    frame = cv2.resize(frame, (640, 480))

    # Process
    detection = detect_with_yolo(frame.copy())
    segmentation = segment_image(frame.copy())
    depth = create_depth_estimation_heatmap(preprocess_image_for_depth_estimation(frame.copy()))
    edges = cv2.cvtColor(apply_canny(frame.copy()), cv2.COLOR_GRAY2BGR)

    # Stack all in a 2x2 grid
    top = np.hstack((detection, segmentation))
    bottom = np.hstack((depth,edges))
    # edge_row = np.hstack((edges, np.zeros_like(edges))) 

    # Combine all
    dashboard = np.vstack((top, bottom))

    # Show the output
    cv2.imshow("Heritage Health Monitoring Dashboard", dashboard)

    # Exit on keypress
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
