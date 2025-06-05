import streamlit as st
import cv2
import time

st.title("Webcam Live Feed")

# Create a placeholder to hold frames
frame_placeholder = st.empty()

# Start/Stop button state using session_state
if "running" not in st.session_state:
    st.session_state.running = False

start_button = st.button("Start", key="start_btn")
stop_button = st.button("Stop", key="stop_btn")

if start_button:
    st.session_state.running = True
if stop_button:
    st.session_state.running = False

if st.session_state.running:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot open webcam")
    else:
        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                st.error("Can't receive frame (stream end?). Exiting...")
                break

            # Convert BGR to RGB for Streamlit display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Show frame in the placeholder
            frame_placeholder.image(frame)

            # Small delay to limit frame rate
            time.sleep(1)

            # Check if user pressed Stop during the loop
            if not st.session_state.running:
                break
        cap.release()
else:
    st.write("Press **Start** to begin capturing from webcam.")
