# 🚧 Automated Crack Detection and Depth Estimation

## 📖 Description

This project presents an **automated defect detection system** focused on crack detection and depth estimation using a combination of computer vision techniques and deep learning models.

Traditional inspection methods for infrastructure are often slow, labor-intensive, and error-prone. While some existing solutions address issues like spalling and discoloration, they typically ignore vegetation interference or lack segmentation capabilities.

This solution uses **YOLO-based segmentation** models and **Canny edge detection** to enhance crack localization. By integrating this into a **real-time interface using Streamlit or OpenCV**, the system enables early crack detection, reducing inspection time and improving maintenance efficiency. When connected with live CCTV or drone footage, the system enables faster, more accurate corrective actions.

## 🚀 Features

- Real-time crack detection via webcam
- Crack segmentation using YOLOv8
- Edge localization using Canny Edge Detection
- Depth estimation (for severity indication)
- Web app interface (Streamlit) and local OpenCV UI
- Lightweight and scalable

## 🛠️ Tech Stack

- **Programming Language:** Python
- **Frontend:** Streamlit, OpenCV
- **Deep Learning Models:** YOLOv8 (via Ultralytics)
- **Libraries Used:**
  - `torch`
  - `opencv-python`
  - `ultralytics`
  - `numpy`
  - `PIL (Pillow)`
  - `streamlit`

## 🧪 How to Run

### Option 1: Run with Streamlit (Web Interface)

1. Open the terminal inside the `demo` folder
2. Run the following command:
   
  to run in streamlit
```bash
streamlit run finalwebapp.py 
```
  to run a realtime opencv interface 
```bash
python camera_capture.py 
```
