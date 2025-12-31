"""
Synology Surveillance Station Object Detection Script
Connects to Surveillance Station, reads camera stream, detects objects, saves annotated video.
"""

import cv2
import numpy as np
import time
from datetime import datetime
from synology_api import surveillance_station
import os

# ============== CONFIGURATION ==============
SYNOLOGY_CONFIG = {
    "ip_address": os.environ.get('ip_address'),      # e.g., "192.168.1.100"
    "port": os.environ.get('ip_address'),                    # Default: 5000 (HTTP) or 5001 (HTTPS)
    "username": os.environ.get('username'),
    "password": os.environ.get('password'),
    "secure": False,                   # Set True for HTTPS
    "cert_verify": False,
    "dsm_version": 7,                  # DSM version (6 or 7)
    "otp_code": None                   # 2FA code if enabled
}

CAMERA_ID = 1                          # Camera ID in Surveillance Station
OUTPUT_DIR = "./recordings"            # Output directory for videos
DETECTION_CONFIDENCE = 0.5             # Minimum confidence threshold
RECORD_DURATION = 30                   # Seconds per video clip

# ============== LOAD DETECTION MODEL ==============
def load_model():
    """Load MobileNet SSD model for object detection."""
    # Download these files if you don't have them:
    # https://github.com/chuanqi305/MobileNet-SSD
    prototxt = "MobileNetSSD_deploy.prototxt"
    weights = "MobileNetSSD_deploy.caffemodel"
    
    net = cv2.dnn.readNetFromCaffe(prototxt, weights)
    
    classes = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor"]
    
    return net, classes

# ============== CONNECT TO SURVEILLANCE STATION ==============
def connect_to_synology():
    """Establish connection to Synology Surveillance Station."""
    print("Connecting to Synology Surveillance Station...")
    
    ss = surveillance_station.SurveillanceStation(
        ip_address=SYNOLOGY_CONFIG["ip_address"],
        port=SYNOLOGY_CONFIG["port"],
        username=SYNOLOGY_CONFIG["username"],
        password=SYNOLOGY_CONFIG["password"],
        secure=SYNOLOGY_CONFIG["secure"],
        cert_verify=SYNOLOGY_CONFIG["cert_verify"],
        dsm_version=SYNOLOGY_CONFIG["dsm_version"],
        otp_code=SYNOLOGY_CONFIG["otp_code"]
    )
    
    print("Connected successfully!")
    return ss

def get_camera_stream_url(ss, camera_id):
    """Get the RTSP or MJPEG stream URL for a camera."""
    # Get camera info
    camera_info = ss.camera_info(camera_id)
    
    # Try to get live view path
    live_path = ss.camera_snapshot(camera_id)  # Gets snapshot URL pattern
    
    # Construct stream URL (adjust based on your setup)
    base_url = f"{'https' if SYNOLOGY_CONFIG['secure'] else 'http'}://{SYNOLOGY_CONFIG['ip_address']}:{SYNOLOGY_CONFIG['port']}"
    stream_url = f"{base_url}/webapi/entry.cgi?api=SYNO.SurveillanceStation.VideoStreaming&version=1&method=Stream&cameraId={camera_id}&format=mjpeg&_sid={ss.session_id}"
    
    return stream_url

# ============== OBJECT DETECTION ==============
def detect_objects(frame, net, classes, confidence_threshold):
    """Detect objects in a frame using MobileNet SSD."""
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    
    net.setInput(blob)
    detections = net.forward()
    
    results = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > confidence_threshold:
            class_id = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            
            results.append({
                "class": classes[class_id],
                "confidence": float(confidence),
                "box": (x1, y1, x2, y2)
            })
    
    return results

def draw_detections(frame, detections):
    """Draw bounding boxes and labels on frame."""
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = f"{det['class']}: {det['confidence']:.2f}"
        
        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1)
        
        # Draw label text
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame

# ============== MAIN PROCESSING LOOP ==============
def main():
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load model
    print("Loading detection model...")
    net, classes = load_model()
    
    # Connect to Synology
    ss = connect_to_synology()
    
    # Get stream URL
    stream_url = get_camera_stream_url(ss, CAMERA_ID)
    print(f"Stream URL: {stream_url}")
    
    # Open video stream
    print("Opening video stream...")
    cap = cv2.VideoCapture(stream_url)
    
    if not cap.isOpened():
        # Fallback: Try direct RTSP if available
        print("MJPEG stream failed, trying RTSP...")
        rtsp_url = f"rtsp://{SYNOLOGY_CONFIG['username']}:{SYNOLOGY_CONFIG['password']}@{SYNOLOGY_CONFIG['ip_address']}:554/Sms=1.unicast"
        cap = cv2.VideoCapture(rtsp_url)
    
    if not cap.isOpened():
        print("Error: Could not open video stream")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 15
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    
    print(f"Stream opened: {width}x{height} @ {fps}fps")
    
    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    clip_start_time = time.time()
    clip_number = 0
    
    def start_new_clip():
        nonlocal out, clip_start_time, clip_number
        clip_number += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{OUTPUT_DIR}/detection_{timestamp}_{clip_number:04d}.mp4"
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        clip_start_time = time.time()
        print(f"Recording: {filename}")
        return filename
    
    current_file = start_new_clip()
    
    print("Starting detection loop (press 'q' to quit)...")
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame, reconnecting...")
                time.sleep(1)
                cap = cv2.VideoCapture(stream_url)
                continue
            
            # Detect objects
            detections = detect_objects(frame, net, classes, DETECTION_CONFIDENCE)
            
            # Draw detections on frame
            annotated_frame = draw_detections(frame.copy(), detections)
            
            # Write to video file
            out.write(annotated_frame)
            frame_count += 1
            
            # Log detections
            if detections:
                detected_classes = [d["class"] for d in detections]
                print(f"Frame {frame_count}: Detected {detected_classes}")
            
            # Start new clip if duration exceeded
            if time.time() - clip_start_time >= RECORD_DURATION:
                out.release()
                print(f"Saved clip: {current_file} ({frame_count} frames)")
                frame_count = 0
                current_file = start_new_clip()
            
            # Display frame (optional - comment out for headless)
            cv2.imshow("Surveillance Detection", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nStopping...")
    
    finally:
        # Cleanup
        if out:
            out.release()
        cap.release()
        cv2.destroyAllWindows()
        print("Done!")

if __name__ == "__main__":
    main()